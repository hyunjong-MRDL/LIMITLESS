import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def load_check_file(args):
    check = pd.read_excel(f"data/excel/check.xlsx")
    tumor_size = check['tumor_size'].values
    tumor_check = check['tumor_check'].values
    
    split_ratio = args.split_ratio
    
    for i in range(len(check)):
        if tumor_size[i] >= split_ratio:
            tumor_check[i] = 'large'
        elif tumor_size[i] < split_ratio:
            tumor_check[i] = 'small'
            
    check['tumor_check'] = tumor_check
    
    return check

def make_dataframe(indices, subject, date, modal, split):
    subject_, date_ = [], []
    for i in range(len(indices)):
        subject_.append(subject[indices[i]])
        date_.append(date[indices[i]])
        
    info = pd.DataFrame({
        'subject' : subject_,
        'date' : date_
    })
    print(f"{modal} => {split} Size : {len(info)}")
    info.to_excel(f'data/excel/{modal}_{split}.xlsx', index=False)
        
def export_subject_date(args):
    check = load_check_file(args)
    size = len(check)
    
    subject = check['subject']
    date = check['date']
    tumor_check = check['tumor_check']
    
    # all
    index = [i for i in range(size)]
    all_train, all_test, _, _ = train_test_split(index, index, test_size=args.test_size, random_state=args.seed)
    all_train, all_val ,_, _ = train_test_split(all_train, all_train, test_size=args.val_size, random_state=args.seed)
    
    make_dataframe(all_train, subject, date, 'all', 'train')
    make_dataframe(all_val, subject, date, 'all', 'val')
    make_dataframe(all_test, subject, date, 'all', 'test')
        
    # large
    large_index = []
    for i in range(size):
        if tumor_check[i] == 'large':
            large_index.append(i)
            
    large_train, large_test, _, _ = train_test_split(large_index, large_index, test_size=args.test_size, random_state=args.seed)
    large_train, large_val ,_, _ = train_test_split(large_train, large_train, test_size=args.val_size, random_state=args.seed) 
    
    make_dataframe(large_train, subject, date, 'large', 'train')
    make_dataframe(large_val, subject, date, 'large', 'val')
    make_dataframe(large_test, subject, date, 'large', 'test')
    
    # small
    small_index = []
    for i in range(size):
        if tumor_check[i] == 'small':
            small_index.append(i)
            
    small_train, small_test, _, _ = train_test_split(small_index, small_index, test_size=args.test_size, random_state=args.seed)
    small_train, small_val ,_, _ = train_test_split(small_train, small_train, test_size=args.val_size, random_state=args.seed) 
    
    make_dataframe(small_train, subject, date, 'small', 'train')
    make_dataframe(small_val, subject, date, 'small', 'val')
    make_dataframe(small_test, subject, date, 'small', 'test')