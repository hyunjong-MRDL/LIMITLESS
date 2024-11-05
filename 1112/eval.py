import os
import glob
import shutil
import pandas as pd
import numpy as np
import SimpleITK as sitk
import argparse

from models import build_model
from sklearn.metrics import confusion_matrix

class Evaluator():
    def __init__(self,
                 args,
                 split):
        self.args = args
        self.split = split
        
    def get_subject_date(self):
        info = pd.read_excel(f"./data/excel/{self.split}_test.xlsx")
        subject = info['subject']
        date = info['date']
        
        return subject, date
    
    def get_path(self):
        path = f"E:\\Liver\\Data"
        
        subject, date = self.get_subject_date()
        A, P, D, label = [], [], [], []
        
        size = len(subject)
        
        for i in range(size):
            datapath = os.path.join(path, subject[i], date[i])
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
                    
        return A, P, D, label, subject, date
    
    def get_volume(self, path):
        info = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(info)
        return info, img
    
    def get_model(self, model_path):
        self.model = build_model(self.args)
        self.model.load_weights(model_path)
        
    def get_eval_metrics(self, target, pred):
        pred = pred.flatten()
        target = target.flatten()
        
        cm = confusion_matrix(target, pred)
        TP = cm[1][1] # 실제 1 예측 1
        TN = cm[0][0] # 실제 0 예측 0

        FP = cm[0][1] # 실제 1 예측 0
        FN = cm[1][0] # 실제 0 예측 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall)) # 2 * TP / 2TP + FP + FN
        #f1_score = 2 * TP / (2*TP + FP + FN)
        iou = TP / (TP + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        print(f"TP : {TP} TN : {TN} FP : {FP} FN : {FN}")
        return f1_score, iou, precision, recall, accuracy
        
    def thresholding(self, volume, threshold):
        copy_volume = volume.copy()
        
        copy_volume[copy_volume >= threshold] = 1
        copy_volume[copy_volume <= threshold] = 0
        
        return copy_volume
    
    def normalize_inten(self, img):
        
        center = self.args.HU_minwindow
        window = self.args.HU_maxwindow
        
        max_val = center+window/2
        min_val = center-window/2
        img_norm = np.clip( (img-min_val)/(max_val-min_val),0,1)*250
        
        return img_norm.astype(np.uint8)
    
    def img2patch(self, img):
        px = self.args.patch_x
        py = self.args.patch_y
        pz = self.args.patch_z
        
        nx, ny, nz, nc = img.shape
        
        shift_range_x = np.int32(px / 2)        
        shift_range_y = np.int32(py / 2)
        shift_range_z = np.int32(pz / 2)
        
        img_aug = []
        
        rx = range(0,nx-px+1,shift_range_x)
        ry = range(0,ny-py+1,shift_range_y)
        rz = range(0,nz-pz+1,shift_range_z)

        for ix in rx:
            for iy in ry:
                for iz in rz:
                    
                    img_aug.append(np.squeeze(img[ix:ix+px,iy:iy+py,iz:iz+pz,:]))
                    
        return np.array(img_aug)
    
    def patch2img(self, patch, nx, ny, nz):
        nn, px, py, pz = patch.shape
        
        shift_range_x = np.int(px / 2)
        shift_range_y = np.int(py / 2)
        shift_range_z = np.int(pz / 2)
        
        img = np.zeros( (nx, ny, nz), dtype=float)
        norm = np.zeros( (nx, ny, nz), dtype=float)
        
        patch_ones = np.ones( (px, py, pz), dtype=float)
        
        rx = range(0,nx-px+1,shift_range_x)
        ry = range(0,ny-py+1,shift_range_y)
        rz = range(0,nz-pz+1,shift_range_z)

        ip = 0
        for ix in rx:
            for iy in ry:
                for iz in rz:
                    
                    img[ix:ix+px,iy:iy+py,iz:iz+pz] += patch[ip]
                    norm[ix:ix+px,iy:iy+py,iz:iz+pz] += patch_ones
                    
                    ip += 1
                    
        img = img/norm
        
        return np.array(img)
        
    def detection(self, model_path):
        A, P, D, label, subject, date = self.get_path()
        
        px = self.args.patch_x
        py = self.args.patch_y
        pz = self.args.patch_z
        
        for i in range(len(A)):
            print(f"Subject : {subject[i]}, date : {date[i]}")
            save_data_path = f'./test/{model_path}/{subject[i]}/{date[i]}'
            os.makedirs(save_data_path, exist_ok=True)
        
            Ainfo, Av = self.get_volume(A[i])
            Pinfo, Pv = self.get_volume(P[i])
            Dinfo, Dv = self.get_volume(D[i])
            labelinfo, labelv = self.get_volume(label[i])
            
            nx, ny, nz = Ainfo.GetSize()
            dx, dy, dz = Ainfo.GetSpacing()
            
            sitk.WriteImage(sitk.GetImageFromArray(Av), os.path.join(save_data_path, 'A.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(Pv), os.path.join(save_data_path, 'P.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(Dv), os.path.join(save_data_path, 'D.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(labelv), os.path.join(save_data_path, 'label.nii.gz'))
            
            A = np.array(Av)
            P = np.array(Pv)
            D = np.array(Dv)
            
            A = np.moveaxis(A, 0, -1)
            P = np.moveaxis(P, 0, -1)
            D = np.moveaxis(D, 0, -1)
            
            img = []
            img.append(A)
            img.append(P)
            img.append(D)
            
            img = np.array(img)
            img = np.moveaxis(img, 0, -1)
            
            patch_img = self.img2patch(self.normalize_inten(img))
            pred_patch_img = self.model.predict(patch_img.reshape(patch_img.shape[0], px, py, pz, 3).astype(np.uint8), batch_size=1)
            pred_img = self.patch2img(np.squeeze(pred_patch_img), nx, ny, nz)
            
            pred_img *= 250.0
            
            det_image = sitk.GetImageFromArray(np.moveaxis(np.squeeze(pred_img), -1, 0).astype(np.float32))
            det_image.SetSpacing((dx, dy, dz))
            
            sitk.WriteImage(det_image, os.path.join(f'{save_data_path}', 'target.nii.gz'))
            shutil.copy(f"E:\\Liver\\Data\\{subject[i]}\\{date[i]}\\Seg\\liver.nii.gz", os.path.join(save_data_path, 'liver.nii.gz'))
            
            organ_seg = sitk.ReadImage(os.path.join(save_data_path, 'liver.nii.gz'))
            tumor_seg = sitk.ReadImage(os.path.join(save_data_path, 'target.nii.gz'))
            
            organ_img = sitk.GetArrayFromImage(organ_seg)
            tumor_img = sitk.GetArrayFromImage(tumor_seg)
            
            tumor_img[np.where(organ_img != 1)] = 0
            print(np.min(tumor_img), np.max(tumor_img))
            tumor_img[np.where(tumor_img <= 60)] = 0
            tumor_img = np.clip(tumor_img / 120, 0, 1)
            print(np.min(tumor_img), np.max(tumor_img))
            
            tumor = sitk.GetImageFromArray(tumor_img)
            tumor.SetSpacing(tumor_seg.GetSpacing())
            
            f = open(os.path.join(save_data_path, 'metric.txt'), 'w')
            
            for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                thres_img = self.thresholding(tumor_img, threshold)
                dice, iou, precision, recall, acc = self.get_eval_metric(thres_img, label)
                sitk.WriteImage(sitk.GetImageFromArray(thres_img), os.path.join(save_data_path, f'pred_{threshold}.nii.gz'))
                print(f"Threshold {threshold} => dice : {dice:>.3f}, iou : {iou:>.3f}, precision : {precision:>.3f}, recall : {recall:>.3f}, acc : {acc:>.3f}")
                data = f"Threshold {threshold} => dice : {dice:>.3f}, iou : {iou:>.3f}, precision : {precision:>.3f}, recall : {recall:>.3f}, acc : {acc:>.3f}\n"
                f.write(data)
                
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split', default='large', type=str)
    parser.add_argument('--all', default=False, type=bool)
    parser.add_argument('--large', default=False, type=bool)
    parser.add_argument('--small', default=False, type=bool)
    
    parser.add_argument('--patch_x', default=64, type=int)
    parser.add_argument('--patch_y', default=64, type=int)
    parser.add_argument('--patch_z', default=16, type=int)
    
    parser.add_argument('--HU_minwindow', default=45, type=int)
    parser.add_argument('--HU_maxwindow', default=300, type=int)
    
    parser.add_argument('--last_epoch', default=0, type=int)
    
    args = parser.parse_args()
    
    model_path = f"{args.model}_all_{int(args.all)}_large_{int(args.large)}_small_{int(args.small)}_{args.last_epoch}.h5"
    Eval = Evaluator(args,
                     args.split)
    
    Eval.detection(f"./model_parameters/{model_path}")