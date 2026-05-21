import os
import math
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from ema_pytorch import EMA
from .data import LLIEDataset
from torch.optim import Adam
from torchvision import utils
from accelerate import Accelerator
from torch.utils.data import DataLoader
from .utils import exists, has_int_squareroot, cycle, divisible_by
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_folders,
        train_mode_names,
        eval_folders,
        eval_mode_names,
        *,
        train_sid=None,
        eval_sid=None,
        illumination_map='ours',
        data_type=torch.float32,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        max_grad_norm = 1.,
        calculate_psnr = False,
        save_best_and_latest_only = False, 
        resume = False,
        job_name = None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            log_with="wandb"
        )

        self.accelerator.init_trackers(
            project_name='LLIE',  
            init_kwargs={
                "wandb": {
                    "name": job_name, 
                    # "settings": wandb.Settings(start_method="fork"),
                    # "resume": resume
                }
            }
        )
        # model
        self.model = diffusion_model
        self.data_type = data_type
        self.channels = diffusion_model.channels

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        self.ds = LLIEDataset(
            folders=train_folders,
            mode_names=train_mode_names,
            image_size=self.image_size,
            augment_horizontal_flip=augment_horizontal_flip,
            convert_image_to=convert_image_to,
            apply_transforms=True,
            return_target=True,
            return_path=False,
            is_val=False,
            illumination_map=illumination_map,
            data_type=data_type,
            sid=train_sid,
        )
        
        self.eval_ds = LLIEDataset(
            folders=eval_folders,
            mode_names=eval_mode_names,
            image_size=self.image_size,
            augment_horizontal_flip=False,
            convert_image_to=convert_image_to,
            apply_transforms=True,
            return_target=True,
            return_path=False,
            is_val=True,
            illumination_map=illumination_map,
            data_type=data_type,
            sid=eval_sid
        )
                
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        eval_dl = DataLoader(self.eval_ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 4)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.eval_dl = self.accelerator.prepare(eval_dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.calculate_psnr = calculate_psnr

        if save_best_and_latest_only:
            assert calculate_psnr, "`calculate_psnr` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_psnr = -1e10

        self.save_best_and_latest_only = save_best_and_latest_only
        if resume:
            self.load('latest')

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(os.path.join(self.results_folder, f'model-{milestone}.pt'), map_location=device, weights_only=False)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x_img, y_img, i_img = next(self.dl)
                    x_img = x_img.to(device)
                    y_img = y_img.to(device)
                    i_img = i_img.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(y_img, **{'low_light': x_img, 'ill_map': i_img})
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                    

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.log({'train_loss': total_loss, 'Step': self.step})

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            all_images_list = []
                            gt_images = []
                            for it, (x_img, y_img, i_img) in tqdm(enumerate(self.eval_dl), desc='Eval:', unit='it', total=len(self.eval_dl)):
                                x_img = x_img.to(device)
                                y_img = y_img.to(device)
                                i_img = i_img.to(device)

                                all_images_list.append(self.ema.ema_model.sample(batch_size=x_img.shape[0], **{'low_light': x_img, 'ill_map': i_img}))
                                gt_images.append(y_img)
                                x_img.to(device)
                                y_img.to(device)
                                i_img.to(device)
                            
                        all_images = torch.cat(all_images_list, dim = 0)
                        all_gt_images = torch.cat(gt_images, dim = 0)

                        if self.calculate_psnr:
                            psnr = compare_psnr(all_gt_images.detach().cpu().numpy(), all_images.detach().cpu().numpy())
                            accelerator.print(f'psnr_score: {psnr}')
                            accelerator.log({'val_psnr': psnr})
                        if self.save_best_and_latest_only:
                            if self.calculate_psnr and self.best_psnr < psnr:
                                self.best_psnr = psnr
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
