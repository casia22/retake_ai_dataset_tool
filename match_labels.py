#!/disk2/app/conda/anaconda3/bin/python3.9
import json
import os

# define function to tansform string like [1370, 874] to list
def str_to_list(string):
    string = string.replace('[', '').replace(']', '')
    string = string.split(',')
    string = [int(i) for i in string]
    return string



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # init paths arguments
    parser.add_argument('--path', type=str, default='./', help='path to the video')
    parser.add_argument('--output', type=str, default='./', help='path to the output')
    

    # set paths
    base_path = parser.parse_args().path 
    #'/disk2/workspace/csgoai/stone/csgo_ai/sample_data/'
    data_path = base_path + 'data/'
    cleaned_path = base_path + 'cleaned_data/'

    # get folders in data_path
    folders = os.listdir(data_path)
    # get json file paths in each folder of data_path
    json_files = [data_path + folder + '/' + folder + '.json' for folder in folders]
    print('json_files:\n', '\n'.join(json_files))
    # get cleaned txt file paths in cleaned folder
    cleaned_files = [cleaned_path + folder + '.txt' for folder in folders]
    print('txt files with no death: \n', '\n'.join(cleaned_files))

    # keep json file that exists
    json_files = [json_file for json_file in json_files if os.path.exists(json_file)]

    # get the null cleaned files and write to file
    null_cleaned_files = [cleaned_file for cleaned_file in cleaned_files if os.path.getsize(cleaned_file) == 0]
    with open(cleaned_path + 'null_cleaned_files.txt', 'w') as f:
        print('empty text files: \n', '\n'.join(null_cleaned_files))
        print('writing empty text files to file: ', cleaned_path + 'null_cleaned_files.txt')
        f.write('\n'.join(null_cleaned_files))


    # keep cleaned file that exists and has size > 0
    cleaned_files = [cleaned_file for cleaned_file in cleaned_files if os.path.exists(cleaned_file) and os.path.getsize(cleaned_file) > 0]
    print('final txt file with no death: \n', '\n'.join(cleaned_files))
    # print a separator line
    print('\n---------------------\n')

    print('\n---------------------\n')
    print('\n---------------------\n')
    print('\n---------------------\n')
    print('\n---------------------\n')
    # keep json file that has corresponding cleaned file
    json_files = []
    for json_file in cleaned_files:
        json_file = json_file.replace('.txt', '.json').replace('cleaned_data', 'data')
        folder = json_file.split('/')[-1].replace('.json', '')
        json_file = data_path + folder + '/' + folder + '.json'
        json_files.append(json_file) 

    # iterate through json files and cleaned files
    for json_file, cleaned_file in zip(json_files, cleaned_files):
        print('processing json file: \n ', json_file)
        print('processing cleaned file: \n ', cleaned_file)

        # read json file
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        # read cleaned file
        with open(cleaned_file, 'r') as f:
            cleaned_data = f.read().split('\n')
        # iterate through cleaned data and get keys
        keys = []
        for cleaned_line in cleaned_data:
            # get timestamp from line. note the timestamp is 19 digits
            # line eaxmple: ./data/roth-1679837416-919064000/1679838639-681973700.jpg
            timestamp = cleaned_line.split('/')[-1].replace('.jpg', '').replace('.png','').replace('-', '.')
            # convert timestamp to key by removing the all the 0s in the end
            key = timestamp.rstrip('0')
            keys.append(key)

        # get the corresponding json labels
        # key_names=['Button.left','Button.right','scroll',"'w'","'a'","'s'","'d'","'r'","'q'","'e'","'b'",\
        #   "'1'","'2'","'3'","'4'","'5'","'6'","'7'","'8'","'9'","'0'",'key.shif','key.spac','key.ctrl']
        mouse_pos = []
        key_stroke_labels = []
        img_paths = []

        for key in keys:
            # get mouse position
            pos = str_to_list(json_data[key]['mouse_pos'])
            mouse_pos.append([pos[0], pos[1]])
            # get key labels
            action = str_to_list(json_data[key]['action'])
            key_stroke_labels.append(action)
            # get img path
            img_path = data_path +  json_data[key]['img'].replace('/result','')
            img_paths.append(img_path)
        # calculate mouse movement
        mouse_movement = []
        for i in range(len(mouse_pos)):
            if i == len(mouse_pos) - 1:
                mouse_movement.append([0, 0])
                continue
            mouse_movement.append([mouse_pos[i+1][0] - mouse_pos[i][0], mouse_pos[i+1][1] - mouse_pos[i][1]])                    
        # discritize mouse movement
        # find closest in list
        # todo: find a better way to discretize!!!!!!1
        for i in range(len(mouse_movement)):
            mouse_x, mouse_y = mouse_movement[i]
            mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
            mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
            mouse_x = min(mouse_x_possibles, key=lambda x_:abs(x_-mouse_x))
            mouse_y = min(mouse_y_possibles, key=lambda x_:abs(x_-mouse_y))
            mouse_movement[i] = [mouse_x, mouse_y]

        # write to file
        with open(cleaned_file.replace('.txt', '_labelled.csv'), 'w') as f:
            # write header
            f.write('mouse_x,mouse_y,click_left,click_right,scroll,w,a,s,d,r,q,e,b,1,2,3,4,5,6,7,8,9,0,shift,space,ctrl,img_path\n')
            # write data
            for i in range(len(mouse_movement)):
                f.write(','.join([str(x) for x in mouse_movement[i] + key_stroke_labels[i]]) + ',' + img_paths[i] + '\n')
        # print progress
        print('finished processing json file: ', json_file)
        print('writing to file: ', cleaned_file.replace('.txt', '_labelled.csv'))
        print('\n-------------------------------------\n')
