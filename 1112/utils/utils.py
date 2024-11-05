import numpy as np

def data_aug(args, img, label, len_x, len_y, len_z): # img, label : 116개의 랜덤 이미지, 라벨 => len_x:64, len_y:64, len_z:16
    shift_range_x = args.patch_x // 4 - 1 
    shift_range_y = args.patch_y // 4 - 1
    shift_range_z = args.patch_z // 4 - 1# args.patch_z // 2
    
    xs = np.random.randint(0,shift_range_x,img.shape[0])
    ys = np.random.randint(0,shift_range_y,img.shape[0])
    zs = np.random.randint(0,shift_range_z,img.shape[0]) 

    img_aug = []
    label_aug = []
    for i in range(img.shape[0]): # 116
        img_aug.append(img[i, xs[i]:xs[i]+len_x, ys[i]:ys[i]+len_y, zs[i]:zs[i]+len_z , :]) # (i, )
        #print(img[i, xs[i]:xs[i]+len_x, ys[i]:ys[i]+len_y, zs[i]:zs[i]+len_z , :].shape)
        label_aug.append(label[i,xs[i]:xs[i]+len_x,ys[i]:ys[i]+len_y,zs[i]:zs[i]+len_z,:])
    
    img_aug = np.array(img_aug)
    label_aug = np.array(label_aug)
    
    return img_aug, label_aug

def centroid3(img):
    nx, ny, nz = img.shape
    
    imgx = np.sum(np.sum(img, axis=1), axis=1)
    imgy = np.sum(np.sum(img, axis=2), axis=0)
    imgz = np.sum(np.sum(img, axis=0), axis=0)
    denom = np.sum(np.sum(np.sum(img, axis=0), axis=0), axis=0)
    cx = np.sum(np.linspace(0, nx-1, nx)*imgx)/denom
    cy = np.sum(np.linspace(0, ny-1, ny)*imgy)/denom
    cz = np.sum(np.linspace(0, nz-1, nz)*imgz)/denom
    
    return cx, cy, cz

def normalize_inten(args, img):

    center = args.HU_minwindow
    window = args.HU_maxwindow
    
    max_val = center+window/2
    min_val = center-window/2
    img_norm = np.clip( (img-min_val)/(max_val-min_val),0,1)*250
    
    return img_norm.astype(np.uint8)
