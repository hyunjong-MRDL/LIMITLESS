import os
import torch
import argparse 

from utils.seed import seed_everything
from data.load_data import get_data_path
from data.dataset import CustomDataset
from torch.utils.data import DataLoader

from models.Unet import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--liver', default=False, type=bool)

    parser.add_argument('--min_window', default=45, type=int)
    parser.add_argument('--max_window', default=300, type=int)
    parser.add_argument('--patch_x', default=32, type=int)
    parser.add_argument('--patch_y', default=128, type=int)
    parser.add_argument('--patch_z', default=128, type=int)

    args = parser.parse_args()

    seed_everything(args)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Model Training in {device}!")

    train, val = get_data_path(args)
    dataset = CustomDataset(args, train)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle=True)

    a, b, c, d = next(iter(dataloader))

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)

    model = build_model(args)

    

if __name__=='__main__':
    main()