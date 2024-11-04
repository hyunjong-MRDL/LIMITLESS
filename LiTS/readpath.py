import os
import numpy as np

root = "D:/Datasets/Liver/"

def path_dict(root):
    total_dict = {}
    for ID in os.listdir(root):
        date_dict = {}
        for date in os.listdir(f"{root}{ID}/"):
            path_list = []
            for item in os.listdir(f"{root}{ID}/{date}/"):
                if item == "Seg":
                    path_list.append(f"{root}{ID}/{date}/{item}/liver.nii.gz")
                elif ".nii" in item:
                    path_list.append(f"{root}{ID}/{date}/{item}")
            date_dict[date] = path_list
        total_dict[ID] = date_dict
    return total_dict