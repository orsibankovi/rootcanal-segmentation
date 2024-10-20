from torch.utils.data import Dataset
import torch
import cv2
import tifffile
from os import listdir
import numpy as np


class GetDataset(Dataset):
    def __init__(
        self, input_path: str, s: int = 256, tmp_dir: str = None, mode: str = "train"
    ):
        self.rootdir = input_path
        self.tmp_dir = tmp_dir
        self.mode = mode
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
        self, volume_array_path: str, target_array_path: str, idx: int
    ) -> tuple[list, list]:
        volume_array = tifffile.imread(volume_array_path)
        target_array = tifffile.imread(target_array_path)
        inputs = []
        targets = []
        for z, image in enumerate(volume_array):
            if np.max(image) != 0 or z % 1 == 0:
                s = volume_array.shape[0]
                if z not in [0, s - 1]:
                    temp_inp = []
                    for i in range(-1, 2):
                        temp_inp.append(volume_array[z + i])
                    target = (
                        self.resize_image(target_array[z], cv2.INTER_NEAREST) // 255
                    )
                    target = torch.tensor(target, dtype=torch.uint8).unsqueeze(0)
                    batch_input = self.create_batch(temp_inp, cv2.INTER_LINEAR) / 255

                    input_path = (
                        self.tmp_dir
                        + "/input_"
                        + self.mode
                        + "_"
                        + str(idx)
                        + "_"
                        + str(z)
                        + ".npy"
                    )
                    target_path = (
                        self.tmp_dir
                        + "/target_"
                        + self.mode
                        + "_"
                        + str(idx)
                        + "_"
                        + str(z)
                        + ".npy"
                    )

                    np.save(input_path, batch_input)
                    np.save(target_path, target)

                    inputs.append(input_path)
                    targets.append(target_path)

        return inputs, targets

    def load_images(self):
        subfolders = listdir(self.rootdir)
        for i, subfolder in enumerate(subfolders):
            input_ = self.rootdir + "/" + subfolder + "/original.tiff"
            target_ = self.rootdir + "/" + subfolder + "/inverse.tiff"
            volume_, mask_ = self.preprocessing(input_, target_, i)
            self.inputfiles += volume_
            self.targetfiles += mask_

    def __len__(self) -> int:
        return len(self.inputfiles)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        input_ = np.load(self.inputfiles[idx])
        target_ = np.load(self.targetfiles[idx])
        return input_, target_
