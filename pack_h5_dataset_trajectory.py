import os
import csv
import time
import h5py
import numpy as np
from PIL import Image
import psutil
import cv2


def img_preprocess(imgs):
    "input img has size (batch, 1024, 768, 3)"
    "swap (0, 3, 1, 2) axis"
    "center crop to 3 498 824 "
    imgs = np.expand_dims(imgs,axis=0)
    # swap
    #print('imgs.shape = ', imgs.shape)#(180, 1024, 768, 3)
    imgs = np.transpose(imgs, (0, 3, 2, 1)) # (batch, 3, 768, 1024)
    #print('imgs.shape = ', imgs.shape) # (180, 3, 768, 1024)
    # center crop
    _, _,height, width = imgs.shape
    crop_size = (498, 824)

    # Calculate the coordinates of the center of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the coordinates of the top-left corner of the crop box
    crop_x = center_x - crop_size[1] // 2
    crop_y = center_y - crop_size[0] // 2

    # Calculate the coordinates of the bottom-right corner of the crop box
    crop_x2 = crop_x + crop_size[1]
    crop_y2 = crop_y + crop_size[0]
    #print('crop_x = ', crop_x, 'crop_y = ', crop_y, 'crop_x2 = ', crop_x2, 'crop_y2 = ', crop_y2)   
    #print('center_x = ', center_x, 'center_y = ', center_y)
    imgs = imgs[:, :, crop_y:crop_y2, crop_x:crop_x2]
    #print('imgs.shape = ', imgs.shape)
    assert imgs.shape == (imgs.shape[0], 3, 498, 824)
    
    # resize image
    resized_imgs = np.zeros((imgs.shape[0], 3, 150, 280), dtype=np.uint8)
    for i in range(imgs.shape[0]):
            #print('shape of imgs!!!!!:', imgs[i].shape) shape of imgs!!!!!: (3, 498, 824)
            img = cv2.resize(imgs[i].transpose(1,2,0), (280,150)).transpose(2,0,1)
            #print('img shape is :!!!',img.shape)#(3, 150, 280)
            resized_imgs[i] = img#cv2.resize(imgs[i].transpose(1,2,0), (280,150)).transpose(2,0,1) 
            # (280,150,3) into shape (3,150,280)


    assert resized_imgs.shape == (imgs.shape[0], 3, 150, 280)
    resized_imgs = np.squeeze(resized_imgs)
    return resized_imgs

# define a function to add a trajectory to the h5 file
def put_trajectory_to_h5(h5_file, img_array, label_array):
    #print('putting trajectory to h5 file')
    print('img_array length = ', img_array.shape[0], 'label_array length = ', label_array.shape[0])
    # if len < 20 then refuse to add
    if img_array.shape[0] < 20:
        print('trajectory length < 20, return')
        return

    # truncate the trajectory if the length is greater than 1100
    if img_array.shape[0] > 1100:
        img_array = img_array[:1100, :, :, :]
        label_array = label_array[:1100, :]

    # complement the trajectory with zeros if the length is less than 1100
    elif img_array.shape[0] < 1100:  
        print('sample length is {}, appending to 1100'.format(img_array.shape[0]))
        print('img_array.shape[0] = ', img_array.shape[0])
        img_array = np.concatenate((img_array, np.zeros((1100 - img_array.shape[0], 3, 150, 280), dtype=np.uint8)), axis=0)
        print('label_array.shape[0] = ', label_array.shape[0])
        label_array = np.concatenate((label_array, np.zeros((1100 - label_array.shape[0], 26), dtype=np.float32)), axis=0)

    # add the trajectory to the h5 file
    t0 = time.time()
    print('starting saving: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    h5_file['data'].resize((h5_file['data'].shape[0] + 1), axis=0)
    h5_file['data'][-1, :, :, :, :] = img_array
    h5_file['label'].resize((h5_file['label'].shape[0] + 1), axis=0)
    h5_file['label'][-1, :, :] = label_array
    t1 = time.time()
    took_time = t1 - t0
    # transform took_time into hours:mins:secs
    took_time = time.strftime('%H:%M:%S', time.gmtime(took_time))
    print('saved. {} took time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), took_time))

def row_count(path_csv):
    with open(path_csv, 'r') as fileObject:
        count = sum(1 for row in fileObject)
    return count

def get_label_img_array(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        # skip the csv header
        next(reader)
        img_paths = [row[-1] for row in reader]
        # fill the img_array and label_array
        #img_array = np.array([cv2.resize(cv2.imread(img_path), (768, 1024)) for img_path in img_paths], dtype=np.uint8)
        # 上面这种实现方式会导致内存溢出，所以改成下面这种实现方式
        img_array = np.zeros((len(img_paths),3,150,280), dtype=np.uint8)
        for i, img_path in enumerate(img_paths):
            img_array[i, :, :, :] = img_preprocess(cv2.resize(cv2.imread(img_path), (768, 1024)))
            #can not (3,150,280) into shape (280,150,3)
    with open(path, 'r') as f:                   
        reader = csv.reader(f)    
        # skip the csv header    
        next(reader)            
        # get label array   
        label_array = np.array([row[:-1] for row in reader], dtype=np.float32)
    # print current ram usage
    print('ram usage: ', psutil.virtual_memory().percent)
    # estimate the ram usage of the img_array and label_array
    img_array_size = img_array.nbytes / 1024 / 1024
    label_array_size = label_array.nbytes / 1024 / 1024
    print('img_array_size = ', img_array_size, 'MB')
    print('label_array_size = ', label_array_size, 'MB')
    return img_array, label_array


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
    h5_file.create_dataset('data', (0, 1100, 3, 150, 280), maxshape=(None, 1100, 3, 150, 280), dtype=np.uint8, chunks=(1,1100, 3, 150, 280))#, compression='gzip')
    h5_file.create_dataset('label', (0, 1100, 26), maxshape=(None, 1100, 26), dtype=np.float32, chunks= (1,1100,26))#, compression='gzip')
    #note: the trajectory length is 1100, if scenario length is 1000, then the last 100 frames are all zeros

    # read csv file and convert corresponding images and labels to h5 file
    # multiple trajectories might be in one csv file,we segment them with the time interval in img_path
    for csv_file in csv_list:
        t0 = time.time()
        print('preprocessing: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        count = row_count(cleaned_path + csv_file)
        with open(cleaned_path + csv_file, 'r') as f:
            reader = csv.reader(f)
            # skip the csv header
            next(reader)
            img_paths = [row[-1] for row in reader]
            # fill the img_array and label_array
            img_array, label_array = get_label_img_array(cleaned_path + csv_file)
            print('get array took time:',time.strftime('%H:%M:%S', time.gmtime(time.time() - t0)))
            # preprocess the image
            #img_array = img_preprocess(img_array)

            # segment the trajectory based on the time interval in img_path
            # the time interval for segmentation is 2s
            time_list = [int(img_path.split('/')[-1].replace('.jpg','').replace('.png','').split('-')[0]) for img_path in img_paths]
            # get the time interval
            time_gap_list = [time_list[i] - time_list[i-1] for i in range(1, len(time_list))]
            

            # time estimate
            took_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))
            print('done preprocessing: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' took time:{}'.format(took_time))

            # detect the time interval that is greater than 2s and segment the trajectory
            if max(time_gap_list) < 2:
                print('max time gap = ', max(time_gap_list))           
                put_trajectory_to_h5(h5_file, img_array, label_array)
            else:
                for i in range(len(time_gap_list)):
                    if time_gap_list[i] > 2:
                        # add the trajectory to the h5 file
                        put_trajectory_to_h5(h5_file, img_array[:i+1], label_array[:i+1])
                        # remove the trajectory from the list
                        img_array = img_array[i+1:]
                        label_array = label_array[i+1:]
                    if i == len(time_gap_list) - 1:
                        # add the trajectory to the h5 file
                        put_trajectory_to_h5(h5_file, img_array, label_array)

    # close the h5 file
    h5_file.close()
    





