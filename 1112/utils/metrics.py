from tensorflow import keras

def dice_coef_oneclass(y_true, y_pred, smooth):
    # Flatten predictions, preserving the class dimension
    y_pred = keras.backend.batch_flatten(y_pred)
    y_true = keras.backend.batch_flatten(y_true)
    
    class_intersection = keras.backend.sum(y_true * y_pred)
    class_loss = (2. * class_intersection + smooth) / (keras.backend.sum(y_true) + keras.backend.sum(y_pred) + smooth)
        
    return class_loss

def hard_dice_coef(y_true, y_pred):
    y_pred = keras.backend.cast(keras.backend.greater(y_pred, 0.5), 'float32')
    return dice_coef_oneclass(y_true, y_pred, 1e-4)

def DICE_loss(y_true, y_pred):
    return 1 - dice_coef_oneclass(y_true, y_pred, smooth=1)

# ================================================================================================================================ #

def iou_coef_oneclass(y_true, y_pred, smooth):
    # Flatten predictions, preserving the class dimension
    y_pred = keras.backend.batch_flatten(y_pred)
    y_true = keras.backend.batch_flatten(y_true)
    
    class_intersection = keras.backend.sum(y_true * y_pred)
    class_loss = (class_intersection + smooth) / (keras.backend.sum(y_true) + keras.backend.sum(y_pred) - class_intersection + smooth)
        
    return class_loss

def hard_iou_coef(y_true, y_pred):
    y_pred = keras.backend.cast(keras.backend.greater(y_pred, 0.5), 'float32')
    return iou_coef_oneclass(y_true, y_pred, 1e-4)

def IOU_loss(y_true, y_pred):
    y_pred = keras.backend.batch_flatten(y_pred)
    y_true = keras.backend.batch_flatten(y_true)
    
    class_intersection = keras.backend.sum(y_true * y_pred)
    class_loss = (class_intersection + 1) / (keras.backend.sum(y_true) + keras.backend.sum(y_pred) - class_intersection + 1)
    return 1 - class_loss

# ================================================================================================================================ #

def IOU_DICE_loss(y_true, y_pred):
    iou_loss = IOU_loss(y_true, y_pred)
    dice_loss = DICE_loss(y_true, y_pred)
    
    weight1, weight2 = 1, 1
    return weight1 * iou_loss + weight2 * dice_loss