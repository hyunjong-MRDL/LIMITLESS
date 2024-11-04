import torch, torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from basic_setup import seed_everything, CFG
from readpath import path_dict

seed_everything(CFG["SEED"])

class LiverDataset(Dataset):
    def __init__(self, dict, transform=None):
        self.dict = dict
        self.transform = transform
    
    def __len__(self):
        return len(self.dict)
    
    def __get_voxel__(self, path):  # (384, 384, 96)
        voxel = nib.load(path).get_fdata()
        return voxel
    
    def __get_date__(self, ID):
        return list(self.dict[ID].keys())
    
    def __merge_APD__(self, A, P, D):  # Needs improvement
        return (A + P + D) / 3
    
    def __normalize__(self, voxel):
        minimum = np.min(voxel)
        maximum = np.max(voxel)
        return ( (voxel - minimum) / (maximum - minimum) * 255 ).astype(np.uint8)

    def __getitem__(self, index):
        ID = list(self.dict.keys())[index]
        date_list = self.__get_date__(ID)
        path_list = self.dict[ID][date_list[0]]  # Use data from the 1st date ONLY
        A = self.__get_voxel__(path_list[0])
        P = self.__get_voxel__(path_list[3])
        D = self.__get_voxel__(path_list[1])
        label = self.__get_voxel__(path_list[2])
        merged = self.__merge_APD__(A, P, D)

        if self.transform is not None:
            merged = self.transform(merged)

        return merged.unsqueeze(0), torchvision.transforms.ToTensor()(label).unsqueeze(0)

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_ratio = CFG["TEST_PORTION"]
data_dict = path_dict("D:/Datasets/Liver/")
total_dataset = LiverDataset(data_dict, train_transform)
train_set, test_set = random_split(total_dataset, [int(len(total_dataset)*(1-test_ratio)), int(len(total_dataset)*test_ratio)])
train_loader = DataLoader(train_set, batch_size=CFG["BATCH_SIZE"], shuffle=True)
test_loader = DataLoader(test_set, batch_size=CFG["BATCH_SIZE"], shuffle=True)