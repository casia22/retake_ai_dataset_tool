#!/disk2/app/conda/anaconda3/bin/python3.9
import os
import time

import numpy as np
import easyocr,cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager,Lock,Value
import multiprocessing


def image_crop(img,path):
    if img is None:
        print(f'None image at {path}')
    img = cv2.resize(img,(1920,1080))
    r,c = img.shape[:2]
    # 手调的自适应比例，用来提高速度
    r_s,r_e =int(200/1080*r),int(750/1080*r)
    c_s,c_e =int(750/1920*c),int(1100/1920*c)
    img  = img[r_s:r_e,c_s:c_e]
    return img


# 按照置信度过滤
def certain_filter(ocr_li,threshold):
    re = []
    for each in ocr_li:
        # print(each)
        if each[2]>threshold:
            re.append(each[1])
        else:
            continue
    return re

def init_glob_var(t):
    # Initialize pool processes global varibales:
    global start_time
    start_time = t
    
def ocr_images(img_paths,counter,task_len,lock):
    flag_list = []
    reader = easyocr.Reader(['en','ch_sim'], gpu=True)
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        flag = 0
        # 图片裁剪
        alive_img = image_crop(img,img_path) # resize to make it faster
        alive_output = reader.readtext(alive_img) 
        alive_output = certain_filter(alive_output,0.8) # 确定度超参数
        # 判断开始
        for each in alive_output:
            if ("选择一件装备" in each or "Retake" in each or "retake" in each):
                flag = 1
        # 判断结束
        for each in alive_output:
            if ("获胜" in each or "杀死" in each or "伤害" in each or "攻击者" in each or "这一时刻" in each):
                flag = 2
            if ("win"in each or "MVP" in each or "Win" in each or "hit" in each or "Hit" in each):
                flag = 2
            if ("Save this moment" in each or "killed you" in each or "their" in each):
                flag = 2
        flag_list.append(flag)
        # 进度条
        progress = counter.value/task_len.value
        est_time = (time.time() - start_time) / (counter.value + 1) * (task_len.value - i)
        est_time = time.strftime("%H:%M:%S", time.gmtime(est_time))
        with lock:
            counter.value += 1
        print("\r[{0:-<50s}] {1:.1f}% {2:s} {3:d} ".format('#' * int(progress * 50), progress * 100, est_time,counter.value), end='')
        # todo:提取得分
        # todo:提取血量
    return flag_list

def parallel_ocr_images(img_paths, num_workers=5):
    results = None
    with multiprocessing.Manager() as manager:
        counter = manager.Value('i', 0)
        task_len = manager.Value('i', len(img_paths))
        lock = manager.Lock() # 创建一个锁

        img_chunks = np.array_split(img_paths, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_glob_var, initargs=(time.time(),)) as executor:
            futures = [executor.submit(ocr_images, chunk,counter,task_len,lock) for chunk in img_chunks]
            #results = [f.result() for f in tqdm(futures, total=len(futures), desc="OCR in progress")]
            results = [f.result() for f in as_completed(futures)]
        final_flags = []
        for each in results:
            final_flags.extend(each)
    return final_flags


def clean(base_path):
    # calculate the number of files in the folder
    num = len(os.listdir(base_path))
    # calculate video length
    sec = num/20
    min = sec/60
    time_str = time.strftime("%H:%M:%S", time.gmtime(sec))
    print('cleaning: ',base_path,'| num of files:',num,'| video length:',time_str)
    # 输入的是某个video的路径，不是根路径
    alive_pic_list = []
    is_alive = 0
    paths = os.listdir(base_path)
    paths.sort()
    paths = [base_path +'/' + path for path in paths]
    # 进度条的参数
    iter_num = len(paths)
    start_time = time.time()
    # 过滤不是图片的文件
    allowed_exts = ('.png', '.jpg', '.jpeg')
    paths = [path for path in paths if path.endswith(allowed_exts)]

    # 并行处理,生成标记列表,开始代表1结束代表2
    flag_list = parallel_ocr_images(paths)
    
    # 仅仅保留1-2之间的 100002形式的paths
    alive_paths = []
    is_alive = 0
    for i,path in enumerate(paths):    
        if (flag_list[i]==1 or is_alive==1):
            is_alive = 1
            alive_paths.append(path)
        if flag_list[i]==2:
            is_alive = 0
    # 增加is_done和step参数
#     steps = [i for in range(len(alive_paths))]
#     is_dones = [0 for in range(len(alive_paths)-1)] + [1]
            
    return alive_paths#,steps,is_dones

if __name__ == '__main__':
    # recieve input arguments
    import argparse
    parser = argparse.ArgumentParser()
    # init paths arguments
    parser.add_argument('--path', type=str, default='.', help='path to the video')
    parser.add_argument('--output', type=str, default='.', help='path to the output')
    # init other arguments
    parser.add_argument('-n', type=str, default='', help='specify the folder name')
    
    # arguement for only clean the lastest folder, if l is specified, then only clean the lastest folder
    parser.add_argument('-l', type=int, nargs='?', const=True, help='clean the lastest folder')


    args = parser.parse_args()
    path_base = args.path#'/disk2/workspace/csgoai/stone/csgo_ai/sample_data'
    # create output folder
    if not os.path.exists(args.output + '/cleaned_data'):
        os.makedirs(args.output + '/cleaned_data')
    # create data folder
    if not os.path.exists(path_base + '/data'):
        os.makedirs(path_base + '/data')
    # count the number of folders in path_base + '/data'
    num = len(os.listdir(path_base+'/data'))
    print('num of folders:',num)

    # if n is specified, then only clean the n folder
    if args.n != '':
        # data cleaning, get valid pics
        path_cur = path_base+'/data'+'/'+args.n
        alive_pics = clean(path_cur)
        #print(alive_pics)
        # save valid pic to path_base + '/cleaned_data/' + each
        with open(args.output + '/cleaned_data/' + args.n + '.txt', 'w') as f:
            f.write('\n'.join(alive_pics))
        print('\n cleaned to :', args.output + '/cleaned_data/' + args.n + '.txt\n')
        exit(0)

    # if l is specified, then only clean the lastest l folder
    if args.l  is not None:
        # get last l folder
        folders = os.listdir(path_base+'/data')
        folders.sort()
        folders = folders[-args.l:]
        # clean every video in path_base folder
        for each in folders:
            # data cleaning, get valid pics
            path_cur = path_base+'/data'+'/'+each
            alive_pics = clean(path_cur)
            #print(alive_pics)
            # save valid pic to path_base + '/cleaned_data/' + each
            with open(args.output + '/cleaned_data/' + each + '.txt', 'w') as f:
                f.write('\n'.join(alive_pics))
            print('\n cleaned to :', args.output + '/cleaned_data/' + each + '.txt\n')
        exit(0)


    # clean every video in path_base folder
    for each in os.listdir(path_base+'/data'):
        # data cleaning, get valid pics
        path_cur = path_base+'/data'+'/'+each
        alive_pics = clean(path_cur)
        #print(alive_pics)
        # save valid pic to path_base + '/cleaned_data/' + each
        with open(args.output + '/cleaned_data/' + each + '.txt', 'w') as f:
            f.write('\n'.join(alive_pics))
        print('\n cleaned to :', args.output + '/cleaned_data/' + each + '.txt\n')
