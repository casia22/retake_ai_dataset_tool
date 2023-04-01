#!/disk2/app/conda/anaconda3/bin/python3.9
import os
import time

import easyocr,cv2

def image_crop(img):
    #img = cv2.resize(img,(1024,576))
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

def clean(path,reader):
    # calculate the number of files in the folder
    num = len(os.listdir(path))
    # calculate video length
    sec = num/20
    min = sec/60
    time_str = time.strftime("%H:%M:%S", time.gmtime(sec))
    print('cleaning: ',path,'| num of files:',num,'| video length:',time_str)
    # 输入的是某个video的路径，不是根路径
    alive_pic_list = []
    is_alive = 0
    paths = os.listdir(path)
    paths.sort()
    # 进度条的参数
    iter_num = len(paths)
    start_time = time.time()
    
    # 按顺序读取文件
    for i,filename in enumerate(paths):
        # 进度条输出
        progress = i/iter_num
        est_time = (time.time() - start_time) / (i + 1) * (iter_num - i)
        est_time = time.strftime("%H:%M:%S", time.gmtime(est_time))
        print("\r[{0:-<50s}] {1:.1f}% {2:s}".format('#' * int(progress * 50), progress * 100, est_time), end='')
        # 跳过不是图片的文件
        if not filename.endswith(('.png','.jpg')):
            continue
        # get the full path of the file
        path1 = os.path.join(path, filename)
        # read ocr
        # read an image
        img = cv2.imread(path1)
        img = image_crop(img) # resize to make it faster
        output = reader.readtext(img)
        #print(output)
        content = certain_filter(output,0.8)
        # 开始的ocr匹配
        # 规则：
            # 检测到“选择一件装备”
        for each in content:
            if "选择一件装备" in each:
                is_alive = 1
        # 结束的ocr匹配
        # 规则：
        #     检测到“获胜”，"杀死"，“伤害”结束
        for each in content:
            if ("获胜" in each or "杀死" in each or "伤害" in each or "攻击者" in each or "这一时刻" in each):
                is_alive = 0
                
        #print('is_alive',is_alive,'|',path1)
        if is_alive==1:
            alive_pic_list.append(path1)
    return alive_pic_list

# define a parallel function
def clean_parallel(path,reader):
    # clean
    alive_pic_list = clean(path,reader)
    # save valid pic to path_base + '/cleaned_data/' + each                                                          
    with open(args.output + '/cleaned_data/' + each + '.txt', 'w') as f:                                             
        f.write('\n'.join(alive_pic_list))                                                                               
    print('\n cleaned to :', args.output + '/cleaned_data/' + each + '.txt\n')

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
    # arguement for parallel, if p is specified, then run in p parallel to clean each folder
    parser.add_argument('-p', type=int, nargs='?', const=True, help='clean in parallel')

    args = parser.parse_args()
    reader = easyocr.Reader(['ch_sim'], gpu=True)
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
        alive_pics = clean(path_cur,reader)
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
            alive_pics = clean(path_cur,reader)

            #print(alive_pics)
            # save valid pic to path_base + '/cleaned_data/' + each
            with open(args.output + '/cleaned_data/' + each + '.txt', 'w') as f:
                f.write('\n'.join(alive_pics))
            print('\n cleaned to :', args.output + '/cleaned_data/' + each + '.txt\n')
        exit(0)




    # clean every video in path_base folder
    if args.p is None:
        for each in os.listdir(path_base+'/data'):
            # data cleaning, get valid pics
            path_cur = path_base+'/data'+'/'+each
            alive_pics = clean(path_cur,reader)
            #print(alive_pics)
            # save valid pic to path_base + '/cleaned_data/' + each
            with open(args.output + '/cleaned_data/' + each + '.txt', 'w') as f:
                f.write('\n'.join(alive_pics))
            print('\n cleaned to :', args.output + '/cleaned_data/' + each + '.txt\n')
    else:
        # run in parallel
        from multiprocessing import Pool
        #RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
        #https://stackoverflow.com/questions/60577308/runtimeerror-cannot-re-initialize-cuda-in-forked-subprocess-to-use-cuda-with-mul
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        # clean every video in path_base folder
        with Pool(args.p) as p:
            for each in os.listdir(path_base+'/data'):
                # data cleaning, get valid pics
                path_cur = path_base+'/data'+'/'+each
                p.apply_async(clean_parallel, args=(path_cur,reader))
            p.close()
            p.join()








        










