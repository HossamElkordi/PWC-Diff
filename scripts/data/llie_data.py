import os
import torch
from torch import nn
from PIL import Image
from functools import partial
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from .data_utils import exists, divisible_by, convert_image_to_fn, get_sid_data, get_illumination_map


class LLIEDataset(Dataset):
    def __init__(
        self,
        folders,
        mode_names,
        image_size,
        illumination_map='ours',
        augment_horizontal_flip = False,
        convert_image_to = None,
        data_type=torch.float32,
        apply_transforms = True,
        return_target = False,
        return_path = False,
        is_val = False,
        sid = None,
        smid = None
    ):
        super().__init__()
        self.folders = folders
        self.mode_names = mode_names
        self.image_size = image_size
        self.return_path = return_path
        self.return_target = return_target
        self.illumination_map = illumination_map
        self.data_type = data_type
        
        self.x_paths = []
        for fldr, mname in zip(folders, mode_names):
            if fldr and mname:
                self.x_paths.extend(
                    sorted(
                        [
                            os.path.join(fldr, mname[0], fname) 
                            for fname in os.listdir(os.path.join(fldr, mname[0]))
                        ]
                    )
                )
        
        if self.return_target:
            self.y_paths = []
            for fldr, mname in zip(folders, mode_names):
                if fldr and mname:
                    self.y_paths.extend(
                        sorted(
                            [
                                os.path.join(fldr, mname[1], fname) 
                                for fname in os.listdir(os.path.join(fldr, mname[1]))
                            ]
                        )
                    )
        
        if sid is not None:
            sid_x, sid_y = get_sid_data(sid)
            self.x_paths.extend(sid_x)
            if self.return_target:
                self.y_paths.extend(sid_y)
        
        transform_list = [] 
        
        if exists(convert_image_to):
            transform_list.append(
                T.Lambda(partial(convert_image_to_fn, convert_image_to))
            )
        
        if apply_transforms:
            transform_list.append(T.CenterCrop(image_size) if is_val else T.RandomCrop(image_size))
            if not is_val and augment_horizontal_flip:
                transform_list.append(T.RandomHorizontalFlip(p=0.5))
        
        transform_list.extend([
            T.ToImage(),
            T.ToDtype(data_type, scale=True)
        ])
        
        self.transform = T.Compose(transform_list)

    
    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, index):
        x_path = self.x_paths[index]
        x_img = Image.open(x_path)
        
        if self.return_target:
            y_path = self.y_paths[index]
            y_img = Image.open(y_path)
            item = self.transform({'x_img': x_img, 'y_img': y_img})
        else:
            item = self.transform({'x_img': x_img})
            
        if not all([divisible_by(d, 8) for d in item['x_img'].shape[-2:]]):
            for k in item.keys():
                item[k] = T.functional.crop(
                    item[k], 
                    top=0, 
                    left=0, 
                    height=item[k].shape[-2] - item[k].shape[-2] % 8,
                    width=item[k].shape[-1] - item[k].shape[-1] % 8
                )
        
        ill = get_illumination_map(map_name=self.illumination_map, x=item['x_img'], dtype=self.data_type)

        if self.return_target:
            if self.return_path:
                return item['x_img'], item['y_img'], ill, x_path
            return item['x_img'], item['y_img'], ill
        
        if self.return_path:
            return item['x_img'], ill, x_path
        return item['x_img'], ill
