import gc
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models import build_model
from utils.utils import data_aug
from utils.metrics import *

# all train
# large train
# small train
# modal -> modal train

class Trainer():
    def __init__(self,
                 args):
        self.args = args

    def get_model(self):
        self.model = build_model(self.args)
        
    def set_args(self):
        pass

    def get_data(self, modal, split):
        img = np.load(f"./npy/{modal}/{modal}_{split}_img.npy", mmap_mode='r')
        label = np.load(f"./npy/{modal}/{modal}_{split}_label.npy", mmap_mode='r')
        label = np.expand_dims(label, axis=-1).astype(np.float32)
        
        return img, label
    
    def get_ts_model(self, to_model_path):
        self.model = build_model(self.args)
        self.model.compile()
        self.model.load_weight(to_model_path)
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.args.lr,
                                                              amsgrad=True),
                           loss= IOU_loss,
                           metrics= [hard_dice_coef, hard_iou_coef])
    
    def ts_train(self, to, go): # ts_train: teacher student model, to: pretrain, go: fine tuning
        to_model_path = self.train(to, finetuning=True)
        self.get_ts_model(to_model_path)
        self.train(go, finetuning=False)
        
    def train(self, modal, finetuning=False):
        total_train_loss, total_val_loss = [], []
        
        self.get_model()
        train_img, train_label = self.get_data(modal, 'train')
        val_img, val_label = self.get_data(modal, 'val')
        
        epochs = self.args.epochs
        sub_num = self.args.random_n
        sub_index = [np.int32(self.num_sample * s / sub_num) for s in range(sub_num + 1)]
        
        tbEpoch = 0
        min_val_loss = 10000.0
        start_train = time.time()
        for epoch in range(epochs):
            data_list = np.arange(self.num_sample)
            np.random.shuffle(data_list)
            print(f"Start epoch : {epoch + 1} / {self.args.epochs}")
            for sub in range(sub_num):  
                tmp_image, tmp_label = data_aug(self.args, 
                                                train_img[data_list[sub_index[sub]:sub_index[sub+1]],:,:,:,:], 
                                                 train_label[data_list[sub_index[sub]:sub_index[sub+1]],:,:,:,:], 
                                                 self.args.patch_x, self.args.patch_y, self.args.patch_z)
                history = self.model.fit(
                     tmp_image, tmp_label,
                     validation_data = data_aug(self.args,
                                                val_img, val_label,
                                                self.args.patch_x, self.args.patch_y, self.args.patch_z),
                    batch_size = self.args.batch_size,
                    epochs = tbEpoch + 1,
                    initial_epoch = tbEpoch,
                    shuffle=False,
                    verbose=1
                 )
                
                val_loss = np.array(history.history['val_loss'])
                train_loss = np.array(history.history['loss'])
                
                total_train_loss.append(train_loss)
                total_val_loss.append(val_loss)
                
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    model_path = f"./model_parameters/{self.args.model}_all_{int(self.args.all)}_large_{int(self.args.large)}_small_{int(self.args.small)}_{epoch+1}.h5"
                    self.model.save_weights(model_path)
                    print(f"----> Model Weight Save!")
                del(history)
                gc.collect()
                
            tbEpoch+=1
        end_train = time.time()
        
        self.logger.info(f"Training Time : {(end_train - start_train) // 60}min {(end_train - start_train) % 60:>.3f}sec")
        
        if self.args.all:
            plt.title(f'Train Loss ALL')
        elif self.args.large:
            plt.title('Train Loss Large')
        elif self.args.small:
            plt.title('Train Loss Small')
            
        plt.plot(total_train_loss)
        plt.clf()

        if self.args.all:
            plt.title(f'Val Loss ALL')
        elif self.args.large:
            plt.title('Val Loss Large')
        elif self.args.small:
            plt.title('Val Loss Small')
            
        plt.plot(total_val_loss)
        plt.clf()
        
        if finetuning:
            return model_path
        
    def do(self):
        # all
        if self.args.all:
            self.train('all', finetuning=False)
        # large
        elif self.args.large:
         self.train('large', finetuning=False)
        # small
        elif self.args.small:
            self.train('small', finetuning=False)
        
        # pretrain -> finetuning
        elif self.args.finetuning:
            self.ts_train(self.args.to, self.args.go)