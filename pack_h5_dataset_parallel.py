import os
import csv
import time
import h5py
import numpy as np
from PIL import Image
import psutil
import cv2
from concurrent.futures import ProcessPoolExecutor

def load_images(img_paths, img_shape):
    img_array = np.zeros((len(img_paths),img_shape[0],img_shape[1],img_shape[2]), dtype=np.uint8)
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f'None image at: {img_path} \n')
            img = np.zeros(img_shape, dtype=np.uint8)
        img_array[i, :, :, :] = img_preprocess(img)
    return img_array

def parallel_load_images(img_paths, img_shape, num_workers=20):
    img_chunks = np.array_split(img_paths, num_workers)
    results = None
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_images, chunk, img_shape) for chunk in img_chunks]
        results = [f.result() for f in futures]
    return np.concatenate(results, axis=0)

def get_label_img_array(path, img_shape, base_path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        # skip the csv header
        next(reader)
        img_paths = [row[-1] for row in reader]
        img_paths = [base_path + path for path in img_paths]

        # fill the img_array and label_array
    img_array = parallel_load_images(img_paths, img_shape)

    with open(path, 'r') as f:
        reader = csv.reader(f)
        # skip the csv header
        next(reader)
        # get label array
        label_array = np.array([row[:-1] for row in reader], dtype=np.float32)
    # print current ram usage
    #print('ram usage: ', psutil.virtual_memory().percent)
    # estimate the ram usage of the img_array and label_array
    img_array_size = img_array.nbytes / 1024 / 1024
    label_array_size = label_array.nbytes / 1024 / 1024
    print(f'img_array_size = {img_array_size:.2f} MB')
    #print(f'label_array_size = {label_array_size:.2f} MB')
    return img_array, label_array

def img_preprocess(img):
    # resize to 1080p
    img = cv2.resize(img, (1920,1080))
    # center crop
    height, width, _ = img.shape
    crop_size = (500, 800)
    center_x = width // 2
    center_y = height // 2
    crop_x = center_x - crop_size[1] // 2
    crop_y = center_y - crop_size[0] // 2
    crop_x2 = crop_x + crop_size[1]
    crop_y2 = crop_y + crop_size[0]
    img = img[crop_y:crop_y2,crop_x:crop_x2,:]
    img = cv2.resize(img,(200,125))
    # resize for performance
    assert img.shape == img_shape#(125, 200, 3)
    return img

# define a function to add a trajectory to the h5 file
def put_trajectory_to_h5(h5_file, img_array, label_array):
    pre_round =h5_file["data"].shape[0]
    print(f"-------------------------添加第{pre_round+1}回合-{img_array.shape[0]}帧---------------------------")
    # if len < 20 then refuse to add
    if img_array.shape[0] < 20:
        print('回合长度 < 20, 不添加')
        return
    if np.sum(label_array[:,2:]) < 20:
        print('回合行为异常，不添加')
        return


    # truncate the trajectory if the length is greater than 1100
    if img_array.shape[0] > 1100:
        img_array = img_array[:1100, :, :, :]
        label_array = label_array[:1100, :]

    # complement the trajectory with zeros if the length is less than 1100
    elif img_array.shape[0] < 1100:
        print('sample length is {}, appending to 1100'.format(img_array.shape[0]))
        # print('img_array.shape[0] = ', img_array.shape[0])
        img_array = np.concatenate((img_array, np.zeros((1100 - img_array.shape[0], img_array.shape[1],img_array.shape[2],img_array.shape[3]), dtype=np.uint8)), axis=0)
        label_array = np.concatenate((label_array, np.zeros((1100 - label_array.shape[0], label_array.shape[1]), dtype=np.float32)), axis=0)

    # add the trajectory to the h5 file
    h5_file['data'].resize((h5_file['data'].shape[0] + 1), axis=0)
    h5_file['data'][-1, :, :, :, :] = img_array
    h5_file['label'].resize((h5_file['label'].shape[0] + 1), axis=0)
    h5_file['label'][-1, :, :] = label_array



def row_count(path_csv):
    with open(path_csv, 'r') as fileObject:
        count = sum(1 for row in fileObject)
    return count

if __name__ == '__main__':
    # read csv file and convert corresponding images and labels to h5 file
    csv_head = 'mouse_x,mouse_y,click_left,click_right,scroll,w,a,s,d,r,q,e,b,1,2,3,4,5,6,7,8,9,0,shift,space,ctrl,img_path'
    # set paths
    base_path = '/disk3/csgo_ai_data/'
    data_path = base_path + 'data/'
    cleaned_path = base_path + 'cleaned_data/'
    h5_path = base_path + 'h5_data/'
    dataset_name = 'csgo_data_new.h5'
    img_shape = (125, 200, 3)
    label_len = 26
    seg_interval = 4

    # get a list of csv files. csvs are within the cleaned_data folder and end with .csv
    csv_list = [f for f in os.listdir(cleaned_path) if os.path.isfile(os.path.join(cleaned_path, f)) and f.endswith('.csv')]


    # create h5 file
    try:
        h5_file = h5py.File(h5_path + dataset_name, 'w')
        # create dataset
        h5_file.create_dataset('data', (0, 1100, img_shape[0], img_shape[1], img_shape[2]), maxshape=(None, 1100, img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8, chunks=(1,1100, img_shape[0], img_shape[1], img_shape[2]))#, compression='gzip')
        h5_file.create_dataset('label', (0, 1100, label_len), maxshape=(None, 1100, label_len), dtype=np.float32, chunks= (1,1100,label_len))#, compression='gzip')
        #note: the trajectory length is 1100, if scenario length is 1000, then the last 100 frames are all zeros

        # read csv file and convert corresponding images and labels to h5 file
        # multiple trajectories might be in one csv file,we segment them with the time interval in img_path
        for csv_file in csv_list:
            t0 = time.time()
            print(f'==========================开始处理：{csv_file}=======================================')
            print('preprocessing: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            count = row_count(cleaned_path + csv_file)
            with open(cleaned_path + csv_file, 'r') as f:
                reader = csv.reader(f)
                # skip the csv header
                next(reader)
                img_paths = [row[-1] for row in reader]
                # fill the img_array and label_array
                img_array, label_array = get_label_img_array(cleaned_path + csv_file,img_shape, base_path)
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
                if max(time_gap_list) < seg_interval:
                    print(f'max time gap is {max(time_gap_list)}s, saving as complete match')
                    put_trajectory_to_h5(h5_file, img_array, label_array)
                else:
                    print(f'max time gap is {max(time_gap_list)}s, saving as separate match')
                    for i in range(len(time_gap_list)):
                        if time_gap_list[i] >= seg_interval:
                            # add the trajectory to the h5 file
                            print(f'saving separate match at {i} with interval {time_gap_list[i]} len{img_array[:i+1].shape[0]}')
                            put_trajectory_to_h5(h5_file, img_array[:i+1], label_array[:i+1])
                            # remove the trajectory from the list
                            img_array = img_array[i+1:]
                            label_array = label_array[i+1:]
                        if i == len(time_gap_list) - 1:
                            # add the trajectory to the h5 file
                            print(f'saving last separate match at {i} with interval {time_gap_list[i]} len{img_array[:i+1].shape[0]}')
                            put_trajectory_to_h5(h5_file, img_array, label_array)

    except:
        import traceback
        traceback.print_exc()
        h5_file.close()
    # close the h5 file
    h5_file.close()

