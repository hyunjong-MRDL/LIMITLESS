import os
import glob
import tqdm
import pandas as pd
import numpy as np

import SimpleITK as sitk
from utils.utils import *

# crop -> patch
# patch

class DataLoad():
    def __init__(self,
                 args,
                 tumorsize):
        self.args = args
        self.tumorsize = tumorsize
        
    def load_info(self, split):
        info = pd.read_excel(f"data/excel/{self.tumorsize}_{split}.xlsx")
        subject, date = info['subject'], info['date']
        
        return subject, date
    
    def get_volume(self, path):
        img = sitk.ReadImage(path)
        imgv = sitk.GetArrayFromImage(img)
        
        return img, imgv
    
    def load_path(self, split):
        path = f"E:\\Liver\\Data"
        subject, date = self.load_info(split)
        
        A, P, D, label, liver = [], [], [], [], []
        size = len(subject)
        
        for i in range(size):
            datapath = os.path.join(path, str(subject[i]), str(date[i]))
            for file in glob.glob(datapath + f"/*.nii.gz"):
                filename = file.split('\\')[5].split('.')[0]
                if filename == 'A':
                    A.append(file)
                elif filename == 'P':
                    P.append(file)
                elif filename == 'D':
                    D.append(file)
                elif filename == 'label':
                    label.append(file)
                    
            liver.append(os.path.join(datapath, f"Seg\\liver.nii.gz"))
            
        return A, P, D, label, liver
    
    def overlay_liver(self, volume, liver):
        volume[liver == 0] = 0
        return volume
    
    def make_patch_data(self, split):
        A, P, D, label, liver = self.load_path(split)
        
        size = len(A)
        imgs, labels = [], []
        
        patch_x = self.args.patch_x
        patch_y = self.args.patch_y
        patch_z = self.args.patch_z
        print(self.args.usage_liver)
        for i in tqdm.tqdm(range(size)):
            Ainfo, Av = self.get_volume(A[i])
            Pinfo, Pv = self.get_volume(P[i])
            Dinfo, Dv = self.get_volume(D[i])
            labelinfo, labelv = self.get_volume(label[i])
            liverinfo, liverv = self.get_volume(liver[i])
            
            Av = np.moveaxis(Av, 0, -1)
            Pv = np.moveaxis(Pv, 0, -1)
            Dv = np.moveaxis(Dv, 0, -1)
            labelv = np.moveaxis(labelv, 0, -1)
            liverv = np.moveaxis(liverv, 0, -1)
            
            nx, ny, nz = labelv.shape
            
            img = []
            if self.args.usage_liver:
                img.append(self.overlay_liver(normalize_inten(self.args, Av), liver))
                img.append(self.overlay_liver(normalize_inten(self.args, Pv), liver))
                img.append(self.overlay_liver(normalize_inten(self.args, Dv), liver))
                
            else:
                img.append(normalize_inten(self.args, Av))
                img.append(normalize_inten(self.args, Pv))
                img.append(normalize_inten(self.args, Dv))
                
            img = np.array(img)
            img = np.moveaxis(img, 0, -1)
            
            for x in range(nx // patch_x):
                for y in range(ny // patch_y):
                    for z in range(nz // patch_z):
                        patch = img[x * patch_x : (x + 1) * patch_x,
                                    y * patch_y : (y + 1) * patch_y,
                                    z * patch_z : (z + 1) * patch_z,
                                    :]
                        
                        patch_label = labelv[x * patch_x : (x + 1) * patch_x,
                                             y * patch_y : (y + 1) * patch_y,
                                             z * patch_z : (z + 1) * patch_z,
                                             ]
                        
                        if self.args.usage_liver:
                            if np.sum(patch) > self.args.liver_ratio:
                                imgs.append(patch)
                                labels.append(patch_label)
                                
                        else:
                            imgs.append(patch)
                            labels.append(patch_label)
                            
        imgs = np.array(imgs)
        labels = np.array(labels)
        
        np.save(f'./{self.tumorsize}_{split}_patch_{self.args.usage_liver}_img.npy', imgs)
        np.save(f'./{self.tumorsize}_{split}_patch_{self.args.usage_liver}_label.npy', labels)
                        
    
    def make_data(self, split):
        A, P, D, label, liver = self.load_path(split)

        size = len(A)
        imgs, labels = [], []
        
        for i in tqdm.tqdm(range(size)):
            Ainfo, Av = self.get_volume(A[i])
            Pinfo, Pv = self.get_volume(P[i])
            Dinfo, Dv = self.get_volume(D[i])
            labelinfo, labelv = self.get_volume(label[i])
            
            labelv = np.moveaxis(labelv, 0, -1)
            
            nx, ny, nz = labelv.shape
            cx, cy, cz = centroid3(labelv)
            
            cx = int(cx)
            cy = int(cy)
            cz = int(cz)
            
            rx = np.linspace(cx-(self.args.crop_x // 2), cx+(self.args.crop_x // 2 - 1), (self.args.crop_x))-np.minimum(cx-(self.args.crop_x // 2),0)-(np.maximum(nx-1,cx+(self.args.crop_x // 2 - 1))-nx+1)
            ry = np.linspace(cy-(self.args.crop_y // 2), cy+(self.args.crop_y // 2 - 1), (self.args.crop_y))-np.minimum(cy-(self.args.crop_y // 2),0)-(np.maximum(ny-1,cy+(self.args.crop_y // 2 - 1))-ny+1)
            rz = np.linspace(cz-(self.args.crop_z // 2), cz+(self.args.crop_z // 2 - 1), (self.args.crop_z))-np.minimum(cz-(self.args.crop_z // 2),0)-(np.maximum(nz-1,cz+(self.args.crop_z // 2 - 1))-nz+1)
            rx = rx.astype(np.int32)
            ry = ry.astype(np.int32)
            rz = rz.astype(np.int32) 
            
            Av = np.moveaxis(Av, 0, -1)
            Pv = np.moveaxis(Pv, 0, -1)
            Dv = np.moveaxis(Dv, 0, -1)
            
            img = []
            img.append(normalize_inten(self.args, Av))
            img.append(normalize_inten(self.args, Pv))
            img.append(normalize_inten(self.args, Dv))
            
            img = np.array(img)
            img = np.moveaxis(img, 0, -1)
            
            labels.append((labelv[rx[0]:rx[-1]+1,ry[0]:ry[-1]+1,rz[0]:rz[-1]+1]).astype(np.uint8))            
            imgs.append(img[rx[0]:rx[-1]+1,ry[0]:ry[-1]+1,rz[0]:rz[-1]+1,:]) 
            
        imgs = np.array(imgs)
        labels = np.array(labels)
        
        np.save(f'./{self.tumorsize}_{split}_img.npy', imgs)
        np.save(f'./{self.tumorsize}_{split}_label.npy', labels)
        
    def save_data(self, split):
        if self.args.patch_training:
            self.make_patch_data(split)
        else:
            self.make_data(split)
            
    def do(self):
        self.save_data('train')
        self.save_data('val')