import os
import gc
import torch
from tqdm import tqdm
from pathlib import Path
from .utils import exists
from ema_pytorch import EMA
from .data import LLIEDataset
from torchvision import utils
from accelerate import Accelerator
from torch.utils.data import DataLoader


class Generator:
    def __init__(
        self,
        diffusion_model,
        input_folder,
        mode_name,
        output_folder,
        sid=None,
        eval_batch_size=1,
        data_type=torch.float32,
        ckpt='best',
        illumination_map='ours',
        convert_image_to=None,
        mixed_precision_type = 'fp16',
        split_batches = True,
        amp = False,
        ema_update_every = 10,
        ema_decay = 0.995,
        results_folder = './results'
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
        )

        self.model = diffusion_model
        self.channels = diffusion_model.channels


        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)
        
        self.batch_size = eval_batch_size
        self.output_folder = output_folder
        
        self.eval_ds = LLIEDataset(
            folders=[input_folder],
            mode_names=[mode_name],
            image_size=None, augment_horizontal_flip=False,
            convert_image_to=convert_image_to,
            data_type=data_type,
            apply_transforms=False,
            return_path=True,
            is_val=False,
            illumination_map=illumination_map,
            sid=sid
        )

        eval_dl = DataLoader(self.eval_ds, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 4)
        self.eval_dl = self.accelerator.prepare(eval_dl)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.accelerator.device)

        self.results_folder = Path(results_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        self.model = self.accelerator.prepare(self.model)
        self.load(ckpt)
    
    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(os.path.join(self.results_folder, f'model-{milestone}.pt'), map_location=device, weights_only=False)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            

    def generate(self):
        self.ema.ema_model.eval()
        device = self.accelerator.device
        with torch.inference_mode():
            for _, (x_img, i_img, p) in tqdm(enumerate(self.eval_dl), desc='Eval', unit='it', total=len(self.eval_dl)):
                if all([os.path.exists(os.path.join(self.output_folder, f.split(os.path.sep)[-1])) for f in p]):
                    continue
                x_img = x_img.to(device)
                i_img = i_img.to(device)
                inputs = {
                    'low_light': x_img, 'ill_map': i_img
                }
                
                out = self.ema.ema_model.sample(batch_size=x_img.shape[0], disable_tqdm=True, **inputs)
                for f, o in zip(p, out):
                    fname = f.split(os.path.sep)[-1]
                    output_path = os.path.join(self.output_folder, fname)
                    if os.path.exists(output_path):
                        continue
                    utils.save_image(o.detach().cpu(), output_path)
                
                x_img = x_img.to('cpu')
                i_img = i_img.to('cpu')
                out = out.to('cpu')
                del x_img, i_img, out
                torch.cuda.empty_cache()
                gc.collect()
