from torch.utils.data import Dataset
import torch
import cv2
import tifffile
from os import listdir
import numpy as np

class GetDataset(Dataset):
    def __init__(self, input_path: str, s: int = 256):
        self.rootdir = input_path
        self.inputfiles = []
        self.targetfiles = []
        self.s = s
        self.load_images()
        
        
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.s, self.s), interpolation=cv2.INTER_LINEAR)
       
        
    def create_batch(self, image_list: list) -> torch.Tensor:
        for i, img in enumerate(image_list):
            image_list[i] = self.resize_image(img)
        return torch.tensor(np.stack(image_list, axis=0), dtype=torch.uint8)


    def preprocessing(self, volume_array_path: str, target_array_path: str) -> tuple[list, list]:
        volume_array = tifffile.imread(volume_array_path)
        target_array = tifffile.imread(target_array_path)
        inputs = []
        targets = []
        for z, image in enumerate(volume_array):
            if np.max(image) != 0 or z % 3 == 0:
                if z != 0 and z != volume_array.shape[0] - 1:
                    batch = self.create_batch([volume_array[z-1], volume_array[z], volume_array[z+1]]) / 255
                    inputs.append(batch)
                    target = cv2.resize(target_array[z], (self.s, self.s), interpolation=cv2.INTER_NEAREST) // 255
                    targets.append(torch.tensor(np.expand_dims(target, axis=0), dtype=torch.uint8))
        return inputs, targets

        
    def load_images(self):
        subfolders = listdir(self.rootdir)
        for subfolder in subfolders:
            input_ = self.rootdir + '/' + subfolder + '/original.tiff'
            target_ = self.rootdir + '/' + subfolder + '/inverse.tiff'
            volume_, mask_ = self.preprocessing(input_, target_)
            self.inputfiles += volume_
            self.targetfiles += mask_
     
        
    def __len__(self) -> int:
        return len(self.inputfiles)
    
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputfiles[idx], self.targetfiles[idx]