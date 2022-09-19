from cProfile import label
from math import ceil
import shutil
import os
from tqdm import tqdm
from glob import glob
import pickle

import cv2
from numpy import DataSource
import pandas as pd

'''
total data parsing to class/state folder
'''
# dataset_dir = os.path.expanduser(os.path.join('~', 'myungsang/datasets/dacon_anomaly_detection'))
# new_dataset_dir = os.path.join(dataset_dir, 'total')

# train_img_dir = os.path.join(dataset_dir, 'train')
# train_df_path = os.path.join(dataset_dir, 'train_df.csv')
# train_df = pd.read_csv(train_df_path)
# class_list = train_df['class']
# state_list = train_df['state']
# file_name_list = train_df['file_name']

# for cls, state, file_name in tqdm(zip(class_list, state_list, file_name_list)):
#     # print(cls, state, file_name)
#     cls_dir = os.path.join(new_dataset_dir, cls)
#     if not os.path.isdir(cls_dir):
#         os.makedirs(cls_dir)
#     state_dir = os.path.join(cls_dir, state)
#     if not os.path.isdir(state_dir):
#         os.makedirs(state_dir)
#     img_path = os.path.join(state_dir, file_name)
#     origin_img_path = os.path.join(train_img_dir, file_name)
#     shutil.copyfile(origin_img_path, img_path)

'''
total data parsing to train, valid 
'''
# dataset_dir = os.path.expanduser(os.path.join('~', 'myungsang/datasets/dacon_anomaly_detection/total'))
# train_dataset_dir = os.path.expanduser(os.path.join('~', 'myungsang/datasets/dacon_anomaly_detection/data/train'))
# valid_dataset_dir = os.path.expanduser(os.path.join('~', 'myungsang/datasets/dacon_anomaly_detection/data/valid'))

# class_dir_list = sorted(os.listdir(dataset_dir))
# for class_dir in class_dir_list:
#     if not os.path.isdir(os.path.join(train_dataset_dir, class_dir)):
#         os.makedirs(os.path.join(train_dataset_dir, class_dir))
#     if not os.path.isdir(os.path.join(valid_dataset_dir, class_dir)):
#         os.makedirs(os.path.join(valid_dataset_dir, class_dir))

#     # print(class_dir)
#     state_dir_list = sorted(os.listdir(os.path.join(dataset_dir, class_dir)))
#     # print(state_dir_list)

    

#     for state_dir in state_dir_list:
#         if not os.path.isdir(os.path.join(train_dataset_dir, class_dir, state_dir)):
#             os.makedirs(os.path.join(train_dataset_dir, class_dir, state_dir))
#         if not os.path.isdir(os.path.join(valid_dataset_dir, class_dir, state_dir)):
#             os.makedirs(os.path.join(valid_dataset_dir, class_dir, state_dir))

#         file_list = sorted(glob(os.path.join(dataset_dir, class_dir, state_dir, '*.png')))
#         # print(len(file_list))
#         valid_num = ceil(len(file_list)*0.1)
#         # print(valid_num)

#         for idx, file_path in enumerate(file_list):
#             if idx < valid_num:
#                 new_file_path = os.path.join(valid_dataset_dir, class_dir, state_dir, os.path.basename(file_path))
#                 shutil.copyfile(file_path, new_file_path)
#             else:
#                 new_file_path = os.path.join(train_dataset_dir, class_dir, state_dir, os.path.basename(file_path))
#                 shutil.copyfile(file_path, new_file_path)

'''
file name to label dict
label to file name dict
'''
# dataset_dir = os.path.expanduser(os.path.join('~', 'myungsang/datasets/dacon_anomaly_detection/total'))
# name2label = dict()
# label2name = dict()

# class_dir_list = sorted(os.listdir(dataset_dir))

# for class_idx, class_dir in enumerate(class_dir_list):
#     state_dir_list = sorted(os.listdir(os.path.join(dataset_dir, class_dir)))

#     name2label_state = dict()
#     label2name_state = dict()
#     state_idx = 0
#     for state_dir in state_dir_list:
#         name2label_state[state_dir] = [class_idx, state_idx]
#         label2name_state[state_idx] = f'{class_dir}-{state_dir}'
#         state_idx += 1

#     name2label[class_dir] = name2label_state
#     label2name[class_idx] = label2name_state

# print(name2label)
# print(label2name)
# # print(state_idx)

# with open(os.path.join(os.getcwd(), 'dataset/name2label.pkl'), 'wb') as f:
#     pickle.dump(name2label, f)

# with open(os.path.join(os.getcwd(), 'dataset/label2name.pkl'), 'wb') as f:
#     pickle.dump(label2name, f)
