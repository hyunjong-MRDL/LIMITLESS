import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split

path = f"E:\\Liver\\LData"

def data_split(args):
    subject = []

    for s in os.listdir(path):
        subject.append(s)

    train_subject, test_subject, _, _ = train_test_split(subject, subject, test_size=args.test_size, random_state = args.seed)
    train_subject, val_subject, _, _ = train_test_split(train_subject, train_subject, test_size = args.val_size, random_state = args.seed)

    train_info = pd.DataFrame({'Subject' : train_subject})
    test_info = pd.DataFrame({'Subject' : test_subject})
    val_info = pd.DataFrame({'Subject' : val_subject})

    train_info.to_excel(os.path.join(path, f"Train.xlsx"))
    test_info.to_excel(os.path.join(path, f"Test.xlsx"))
    val_info.to_excel(os.path.join(path, f"Val.xlsx"))

    return train_subject, val_subject, test_subject

def get_path(args, subject):
    A, P, D, label = [], [], [], []
    liver = []

    for s in subject:
        subject_path = os.path.join(path, s)
        try:
            for date in os.listdir(subject_path):
                date_path = os.path.join(subject_path, date)
                for file in glob.glob(date_path + f"/*.nii.gz"):
                    filename = file.split('\\')[5].split('.')[0]
                    if filename == 'A':
                        A.append(file)
                    elif filename == 'P':
                        P.append(file)
                    elif filename == 'D':
                        D.append(file)
                    elif filename == 'label':
                        label.append(file)

                if args.liver == True:
                    liver.append(os.path.join(date_path, "Seg", "liver.nii.gz"))
                    
        except NotADirectoryError:
            print(f"It's Excel FILE! That's OKAY!")

    if args.liver == True:
        return A, P, D, label, liver
    else:
        return A, P, D, label

def get_data_path(args):
    train_subject, val_subject, test_subject = data_split(args)

    print(f"Train Subject : {len(train_subject)}\nVal Subject : {len(val_subject)}\nTest Subject : {len(test_subject)}")

    if args.liver == True:
        train_A, train_P, train_D, train_label, train_liver = get_path(args, train_subject)
        val_A, val_P, val_D, val_label, val_liver = get_path(args,val_subject)

        print(f"Train phase => (A) : {len(train_A)} (P) : {len(train_P)} (D) : {len(train_D)} (label) : {len(train_label)} (liver) : {len(train_liver)}")
        print(f"Val phase => (A) : {len(val_A)} (P) : {len(val_P)} (D) : {len(val_D)} (label) : {len(val_label)} (liver) : {len(val_liver)}")

        return [train_A, train_P, train_D, train_label, train_liver],\
                [val_A, val_P, val_D, val_label, val_liver]
    else:
        train_A, train_P, train_D, train_label = get_path(args,train_subject)
        val_A, val_P, val_D, val_label = get_path(args,val_subject)

        print(f"Train phase => (A) : {len(train_A)} (P) : {len(train_P)} (D) : {len(train_D)} (label) : {len(train_label)}")
        print(f"Val phase => (A) : {len(val_A)} (P) : {len(val_P)} (D) : {len(val_D)} (label) : {len(val_label)}")

        return [train_A, train_P, train_D, train_label],\
                [val_A, val_P, val_D, val_label]