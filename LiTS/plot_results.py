import numpy as np
import matplotlib.pyplot as plt

def train_plot(loss_hist, iou_hist):
    plt.figure(figsize=(20,10))
    plt.subplot(121), plt.plot(loss_hist, label='train_loss')
    plt.title('Train Loss')

    plt.subplot(122), plt.plot(iou_hist, label='train_iou')
    plt.title('Train IOU')
    plt.savefig("./Figures/Train.jpg")
    return

def test_plot(pred, gt):
    from utils import calculate_iou
    iou_scores = calculate_iou(pred, gt)
    plt.figure(figsize=(20,10))
    plt.plot(iou_scores, label='test_iou')
    plt.title('Train Loss')
    plt.savefig("./Figures/Test.jpg")
    return