"""
    Divide the SceneFlow dataset into train and test subsets.
    Subset flying things 3D has been divided, using theses two partitions directly.
    For the other two subset, randomly assign 1/6 of each folder to the test subset, 
    remaining images form the train subset.
    
    Save the paritition information in a pickle.
    For each subset, store a list of record,
    for each record, has two key-value pairs:
        relative_dir: the relative folder path in the png/disparity folder
        img_list: the img filenames in that relative_dir
"""
import os, sys
import pickle

png_root = "/home/laoreja/dataset/SceneFlow/frames_cleanpass_png"
fly_train_root = "/home/laoreja/dataset/SceneFlow/frames_cleanpass_png/TRAIN"
fly_test_root = "/home/laoreja/dataset/SceneFlow/frames_cleanpass_png/TEST"

pickle_path = '/home/laoreja/dataset/SceneFlow/train_test_list.pickle'

train_cnt = 0
test_cnt = 0
train_list = []
test_list = []

def recursive(root, relative_root, root_parent):
    contents = os.listdir(root)
    if contents[0].endswith('.png'): # is last directory
        global train_cnt
        global test_cnt
        global train_list
        global test_list
        
        test_end = len(contents) / 6
        train_record = {'relative_dir':root_parent, 'img_list':[]}
        test_record = {'relative_dir':root_parent, 'img_list':[]}
        
        for i in xrange(test_end):
            test_record['img_list'].append(contents[i])
        test_cnt += test_end
        test_list.append(test_record)
        for i in xrange(test_end, len(contents)):
            train_record['img_list'].append(contents[i])
        train_cnt += (len(contents) - test_end)
        train_list.append(train_record)        
#        for img_name in contents:
#            if not img_name.endswith('.png'): # All are png files
#                print root, img_name
#                continue
        return
    else:
        for subdir in contents:
            if subdir == 'TEST' or subdir == 'TRAIN' or subdir == 'right':
                continue
            recursive(os.path.join(root, subdir), os.path.join(relative_root, subdir) if relative_root != '' else subdir, relative_root)
            
def recursive_w_mode(root, relative_root, root_parent, mode):
    contents = os.listdir(root)
    if contents[0].endswith('.png'): # is last directory
        record = {'relative_dir':root_parent, 'img_list':[]}
        for i in xrange(len(contents)):
            record['img_list'].append(contents[i])
            
        if mode == 'TRAIN':
            global train_cnt
            global train_list
            train_cnt += len(contents)
            train_list.append(record)
        elif mode == 'TEST':
            global test_cnt
            global test_list
            test_cnt += len(contents)
            test_list.append(record)
        return
    else:
        for subdir in contents:
            if subdir == 'right':
                continue
            recursive(os.path.join(root, subdir), os.path.join(relative_root, subdir) if relative_root != '' else subdir, relative_root)
        
recursive(png_root, '', None)
print train_cnt, test_cnt
print train_list[-1]
print test_list[-1]

recursive_w_mode(fly_train_root, 'TRAIN', '', 'TRAIN')
recursive_w_mode(fly_test_root, 'TEST', '', 'TEST')

print train_cnt, test_cnt
print train_list[-1]
print test_list[-1]

with open(pickle_path, 'w') as fd:
    pickle.dump((train_cnt, test_cnt, train_list, test_list), fd)

