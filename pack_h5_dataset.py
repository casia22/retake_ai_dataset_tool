import os
import csv
import h5py
import numpy as np
from PIL import Image


def img_resize_centercrop(img):
    "original img has size (1080, 1920, 3)"
    "resize to 768 1024 3"
    "center crop to 498 824 3"
    img = Image.fromarray(img)
    img = img.resize((1024, 768), Image.ANTIALIAS)
    img = np.array(img)
    img = img[135:633, 98:922, :]
    return img

if __name__ == '__main__':
    # read csv file and convert corresponding images and labels to h5 file
    csv_head = 'mouse_x,mouse_y,click_left,click_right,scroll,w,a,s,d,r,q,e,b,1,2,3,4,5,6,7,8,9,0,shift,space,ctrl,img_path'
    # set paths
    base_path = '/disk2/workspace/csgoai/stone/csgo_ai/sample_data/'
    data_path = base_path + 'data/'
    cleaned_path = base_path + 'cleaned_data/'
    h5_path = base_path + 'h5_data/'
    # get a list of csv files. csvs are within the cleaned_data folder and end with .csv
    csv_list = [f for f in os.listdir(cleaned_path) if os.path.isfile(os.path.join(cleaned_path, f)) and f.endswith('.csv')]

    # create h5 file
    h5_file = h5py.File(h5_path + 'csgo_data.h5', 'w')
    # create dataset
    h5_file.create_dataset('data', (0, 3, 498, 824), maxshape=(None, 3, 498, 824), dtype=np.float32)
    h5_file.create_dataset('label', (0, 26), maxshape=(None, 26), dtype=np.float32)
    # (26,) -> (1, 32)
    # read csv file and convert corresponding images and labels to h5 file
    for csv_file in csv_list:
        with open(cleaned_path + csv_file, 'r') as f:
            reader = csv.reader(f)
            # skip the csv header
            next(reader)
            for row in reader:
                # get the image path
                img_path = row[-1]
                # read the image
                img = Image.open(img_path)
                # convert to numpy array
                img = np.array(img)
                # resize and center crop the image
                img = img_resize_centercrop(img)
                # add a new dimension to the image
                img = np.expand_dims(img, axis=0)
                # convert to float32
                img = img.astype(np.float32)
                # normalize the image
                img /= 255.0
                # transpose the image
                img = np.transpose(img, (0, 3, 1, 2))
                # get the label
                label = np.array(row[:-1], dtype=np.float32)
                # add the image and label to the h5 file
                h5_file['data'].resize(h5_file['data'].shape[0] + 1, axis=0)
                h5_file['data'][-1:] = img
                h5_file['label'].resize(h5_file['label'].shape[0] + 1, axis=0)
                h5_file['label'][-1:] = label
    # close the h5 file
    h5_file.close()
    





