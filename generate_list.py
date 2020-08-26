'''
Created on Mon Jun 25 17:28:04 2018
This file use to generate the lists containing the image paths and labels of left and right eyes respectively,
using for training and valing.

@author: Dragon
'''

from sklearn.utils import shuffle

# 训练集/验证集划分比例
val_size = 0.2
# 图片文件夹路径
doc_path = '/shared_folders/train_data/'

csv_lines = open('./list/trainLabels_referable.csv').readlines()[1:]
print('# total images: {}'.format(len(csv_lines)))

train_lines_left = []
val_lines_left = []
train_lines_right = []
val_lines_right = []
dic_left_name = {}
dic_right_name = {}
label_00 = []
label_11 = []
label_01 = []
label_10 = []
dataset = []

for line in csv_lines:
    image_name, label = line.strip().split(',')
    # 按【标签：名字】构建左眼数据字典
    if image_name[-4] == 'l':
        dic_left_name[image_name] = label
    # 按【名字：标签】构建右眼数据字典
    if image_name[-5] == 'r':
        dic_right_name[image_name] = label

for left_name, right_name in zip(dic_left_name.keys(),dic_right_name.keys()):
    left_label = dic_left_name[left_name]
    right_label = dic_right_name[right_name]
    if left_label == right_label == '0':
        label_00.append(left_name[:-4])
    elif left_label == right_label == '1':
        label_11.append(left_name[:-4])
    elif left_label == '0':
        label_01.append(left_name[:-4])
    elif left_label == '1':
        label_10.append(left_name[:-4])

for data, label in zip([label_00,label_11,label_01,label_10], ['00','11','01','10']):
    data = shuffle(data)
    num_val = int(len(data) * val_size)
    print('eyes with type {} and {} respectively: {} pairs'.format(label[0], label[1], len(data)))
    for image_name in data[:num_val]:
        to_write_left = doc_path + '{}left.jpeg {}\n'.format(image_name, label[0])
        to_write_right = doc_path + '{}right.jpeg {}\n'.format(image_name, label[1])
        val_lines_left.append(to_write_left)
        val_lines_right.append(to_write_right)
    for image_name in data[num_val:]:
        to_write_left = doc_path + '{}left.jpeg {}\n'.format(image_name, label[0])
        to_write_right = doc_path + '{}right.jpeg {}\n'.format(image_name, label[1])
        train_lines_left.append(to_write_left)
        train_lines_right.append(to_write_right)

#将左右眼训练集按相同随机状态打乱
train_lines_left = shuffle(train_lines_left, random_state = 9527)
train_lines_right = shuffle(train_lines_right, random_state = 9527)
#将左右眼验证集按相同随机状态打乱
val_lines_left = shuffle(val_lines_left, random_state = 2333)
val_lines_right = shuffle(val_lines_right, random_state = 2333)

print('# train of left: {}, # train of right: {}\n# val of left: {}, # val of right: {}' \
      .format(len(train_lines_left), len(train_lines_right), len(val_lines_left), len(val_lines_right)))

with open('./list/kaggle_train_left.list', 'w') as f_train_left:
    for train_line in train_lines_left:
        f_train_left.write(train_line)

with open('./list/kaggle_train_right.list', 'w') as f_train_right:
    for train_line in train_lines_right:
        f_train_right.write(train_line)

with open('./list/kaggle_val_left.list', 'w') as f_val_left:
    for val_line in val_lines_left:
        f_val_left.write(val_line)

with open('./list/kaggle_val_right.list', 'w') as f_val_right:
    for val_line in val_lines_right:
        f_val_right.write(val_line)

print('kaggle_train.list and kaggle_val.list generated!')
