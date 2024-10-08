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

    def resize_image(self, image: np.ndarray, interpolation) -> np.ndarray:
        return cv2.resize(image, (self.s, self.s), interpolation=interpolation)

    def create_batch(self, image_list: list, interpolation) -> torch.Tensor:
        for i, img in enumerate(image_list):
            image_list[i] = self.resize_image(img, interpolation)
        return torch.tensor(np.stack(image_list, axis=0), dtype=torch.uint8)

    def preprocessing(
        self, volume_array_path: str, target_array_path: str
    ) -> tuple[list, list]:
        volume_array = tifffile.imread(volume_array_path)
        target_array = tifffile.imread(target_array_path)
        inputs = []
        targets = []
        for z, image in enumerate(volume_array):
            if np.max(image) != 0 or z % 1 == 0:
                s = volume_array.shape[0]
                if z not in [0, 1, s - 2, s - 1]:
                    temp_inp = []
                    temp_target = []
                    for i in range(-2, 3):
                        temp_inp.append(volume_array[z + i])
                        temp_target.append(target_array[z + i])
                    batch_inp = self.create_batch(temp_inp, cv2.INTER_LINEAR) / 255
                    inputs.append(batch_inp)
                    batch_target = (
                        self.create_batch(temp_target, cv2.INTER_NEAREST) // 255
                    )
                    targets.append(batch_target)
        return inputs, targets

    def load_images(self):
        subfolders = listdir(self.rootdir)
        for subfolder in subfolders:
            input_ = self.rootdir + "/" + subfolder + "/original.tiff"
            target_ = self.rootdir + "/" + subfolder + "/inverse.tiff"
            volume_, mask_ = self.preprocessing(input_, target_)
            self.inputfiles += volume_
            self.targetfiles += mask_

    def __len__(self) -> int:
        return len(self.inputfiles)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputfiles[idx], self.targetfiles[idx]
