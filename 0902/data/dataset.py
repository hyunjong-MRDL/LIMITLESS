import torch
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,
                 args,
                 path):
        self.args = args
        self.A = path[0]
        self.P = path[1]
        self.D = path[2]
        self.label = path[3]

    def __len__(self):
        return len(self.A)
    
    def adjust_hounsfield_unit(self, volume):
        min_window = self.args.min_window
        max_window = self.args.max_window

        copy_volume = volume.copy()

        copy_volume[copy_volume <= min_window] = min_window
        copy_volume[copy_volume >= max_window] = max_window

        return copy_volume
    
    def read_volume(self, path):
        volume = sitk.GetArrayFromImage(sitk.ReadImage(path))
        return volume
    
    def patchwise_volume(self, volume):
        patch_volume = []

        C, H, W = volume.shape

        patch_x = self.args.patch_x
        patch_y = self.args.patch_y
        patch_z = self.args.patch_z

        for i in range(C // patch_x):
            for j in range(H // patch_y):
                for z in range(W // patch_z):
                    patch_volume.append(
                        volume[i * patch_x : (i+1) * patch_x,
                               j * patch_y : (j+1) * patch_y,
                               z * patch_z : (z+1) * patch_z]
                    )

        return np.array(patch_volume)
    
    def normalize_inten(self, volume):
        copy_volume = volume.copy()

        norm_volume = np.clip( (copy_volume - self.args.min_window) / (self.args.max_window - self.args.min_window), 0, 1 )

        return norm_volume
    
    def __getitem__(self, index):
        volume_A = self.normalize_inten(self.adjust_hounsfield_unit(self.read_volume(self.A[index])))
        volume_P = self.normalize_inten(self.adjust_hounsfield_unit(self.read_volume(self.P[index])))
        volume_D = self.normalize_inten(self.adjust_hounsfield_unit(self.read_volume(self.D[index])))

        patch_A = self.patchwise_volume(volume_A)
        patch_P = self.patchwise_volume(volume_P)
        patch_D = self.patchwise_volume(volume_D)
        
        label = self.read_volume(self.label[index])

        assert np.all((label == 0) | (label == 1))

        return torch.from_numpy(patch_A),\
                torch.from_numpy(patch_P),\
                torch.from_numpy(patch_D),\
                torch.from_numpy(label).unsqueeze(0)