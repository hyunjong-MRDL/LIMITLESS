import argparse

from data.load_data import DataLoad
from data.split_large_small import export_subject_date

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    
    # Data Argument
    parser.add_argument('--test_size', default=0.1, type=float)
    parser.add_argument('--val_size', default=0.1, type=float)
    
    parser.add_argument('--split_ratio', default=400, type=int, help='The criterion tumor size for split large and small')
    
    parser.add_argument('--all', default=False, type=bool)
    parser.add_argument('--large', default=False, type=bool)
    parser.add_argument('--small', default=False, type=bool)
    
    parser.add_argument('--usage_liver', default=False, type=bool)
    parser.add_argument('--liver_ratio', default=0.0, type=float)
    parser.add_argument('--patch_training', default=False, type=bool)
    
    parser.add_argument('--crop_x', default=128, type=int)
    parser.add_argument('--crop_y', default=128, type=int)
    parser.add_argument('--crop_z', default=32, type=int)
    
    parser.add_argument('--patch_x', default=64, type=int)
    parser.add_argument('--patch_y', default=64, type=int)
    parser.add_argument('--patch_z', default=16, type=int)
    
    parser.add_argument('--HU_minwindow', default=45, type=int)
    parser.add_argument('--HU_maxwindow', default=300, type=int)
    
    # Model
    parser.add_argument('--input_size', default=(64, 64, 16), type=int)
    parser.add_argument('--filter_num', default=[16, 32, 64], type=list)
    parser.add_argument('--out_channel', default=1, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--patch_stride', default=4, type=int)
    parser.add_argument('--stack_num_up', default=2, type=int)
    parser.add_argument('--stack_num_down', default=2, type=int)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--num_mlp', default=2048, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_transformer', default=4, type=int)
    
    # Training
    parser.add_argument('--to', default='_', type=str)
    parser.add_argument('--go', default='_', type=str)
    parser.add_argument('--finetuning', default=False, type=bool)
    parser.add_argument('--model', default='TransUNet', type=str)
    
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=2000, type=int)
    
    parser.add_argument('--random_n', default=5, type=int)
    
    
    args = parser.parse_args()
    
    # make subject & date , split larg and small size
    export_subject_date(args)
    
    if args.all:
        dataloader = DataLoad(args,
                              'all')
    elif args.large:
        dataloader = DataLoad(args,
                              'large')
    elif args.small:
        dataloader = DataLoad(args,
                              'small')
    # Make data & Save data
    dataloader.do()
    
    
    
if __name__=='__main__':
    main()