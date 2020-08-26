# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:30:40 2018

@author: Dragon
"""
from sklearn.metrics import roc_auc_score, roc_curve
from inception_v3 import InceptionV3
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import GlobalAveragePooling2D, Conv2D, BatchNormalization, Lambda
from keras.layers import Dense, Concatenate, Dropout, Input,ActivityRegularization
from keras.models import Model, load_model
from keras import backend as K
from batch_generator import generator_img_batch
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, CSVLogger
from my_callbacks import LossHistory, RocAucMetric, LRWithWarmRestart
from sklearn.utils import class_weight
from math import ceil
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
# import shapely.geometry as SG

#---------------------------------常量设置--------------------------------------
dropout_keep_prob = 0.5
LIST_ROOT = './list'
LEARNING_RATE = 0.0005
NBR_EPOCHS = 3
BATCH_SIZE = 5
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'loss'
NBR_CLASSES = 1
USE_CLASS_WEIGHTS = False
GPUS = "3"
MAX_Q_SIZE = 160
WORKERS = 1
FINE_TUNE = False

# train_data_lines_right = open('./list/kaggle_train_right1.list').readlines()
#-----------------------------构造Siamese网络模型-------------------------------
# select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS

'''
if FINE_TUNE:
    print('Finetune and Loading the Best Model ...')
    model = load_model("./best_weight/siamese_inception.h5")

    # for layer in model.layers:
    #     layer.trainable = True
    #
    # # 指定优化器
    # optimizer1 = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0, nesterov=False)
    # optimizer2 = RMSprop(lr=LEARNING_RATE)
    # optimizer3 = RMSprop()
    #
    # # 编译模型
    # model.compile(optimizer=optimizer3,
    #               loss={'left_output': 'binary_crossentropy', 'right_output': 'binary_crossentropy'},
    #               metrics=['accuracy'])

else:
    # 创建预训练模型
    base_model = InceptionV3(weights='./pre_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                             include_top=False, pooling='avg', model_name='left_inception_v3')

    # base_model2 = InceptionV3(weights='./pre_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                          include_top=False, pooling='avg', model_name='right_inception_v3')
    # 左右眼输入
    input_shape = (299, 299, 3)
    left_input = Input(input_shape, name='left_input')
    right_input = Input(input_shape, name='right_input')

    # left_x = base_model1(left_input)
    # right_x = base_model2(right_input)

    # inception_resnet的bottle_neck输出
    # with tf.variable_scope("Inception", reuse=None):
    #     left_x = base_model(left_input)
    # with tf.variable_scope("Inception", reuse=True):
    #     right_x = base_model(right_input)

    with tf.variable_scope("Inception") as scope:
        left_x = base_model(left_input)
        scope.reuse_variables()
        right_x = base_model(right_input)

    # 使用Concatenate组合
    branches = [left_x, right_x]
    x = Concatenate(axis=-1, name='Siamese_Concatenate')(branches)
    # x = Conv2D(1536, 3, use_bias= False, name='Conv2d_8a_3x3')(x)
    # x = BatchNormalization(axis=3, scale=False, name='Conv2d_8a_BatchNorm')(x)
    x = Dense(512, activation='relu',name='Dense_512')(x)
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout_Final')(x)

    # sigmoid预测分类
    left_output = Dense(1, activation='sigmoid', name='left_output')(x)
    right_output = Dense(1, activation='sigmoid', name='right_output')(x)

    # 实例化模型
    model = Model(inputs=[left_input,right_input], outputs=[left_output,right_output])
    # model.summary()

    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name, layer.trainable)

    # 只训练新加的Top层，冻结InceptionResnetV2所有层(83)
    for layer in base_model.layers:
        layer.trainable = True
    # for layer in base_model.layers[-83:]:
    #     layer.trainable = True

    #指定优化器
    optimizer1 = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0, nesterov = True)
    optimizer2 = RMSprop(lr=LEARNING_RATE)
    optimizer3 = Adam()

    # 编译模型
    model.compile(optimizer=optimizer1,
                  loss={'left_output':'binary_crossentropy','right_output':'binary_crossentropy'},
                  metrics=['accuracy'])

    print('Compiled successfully...')

#------------------------------定制Callback-----------------------------------------
# 自动保存最佳模型
best_model_file = './best_weight/siamese_inception_test.h5'
# 定义几个callback
# best_model = ModelCheckpoint(best_model_file, monitor='val_' + monitor_index,
#                             verbose = 1, save_best_only = True)

reduce_lr = ReduceLROnPlateau(monitor='val_' + monitor_index, factor=0.5, patience=5, verbose=1, min_lr=0.0000001)

# reduce_lr = LRWithWarmRestart(monitor='val_' + monitor_index, factor=0.2, patience=2, min_lr=0.0000001, auto=False)

# early_stop = EarlyStopping(monitor='auc', patience=15, verbose=1)

loss_curve = LossHistory()

result_save = CSVLogger('./list/result.csv', append=True)

#--------------------------------开始训练---------------------------------------
#读入左眼训练集路径
train_path = os.path.join(LIST_ROOT, 'kaggle_train_left1.list')
train_data_lines_left = open(train_path).readlines()
# Check if image path exists.
train_data_lines_left = [w for w in train_data_lines_left if os.path.exists(w.strip().split(' ')[0])]
train_labels_left = [int(w.strip().split(' ')[-1]) for w in train_data_lines_left]
#读入右眼训练集路径
train_path = os.path.join(LIST_ROOT, 'kaggle_train_right1.list')
train_data_lines_right = open(train_path).readlines()
# Check if image path exists.
train_data_lines_right = [w for w in train_data_lines_right if os.path.exists(w.strip().split(' ')[0])]
train_labels_right = [int(w.strip().split(' ')[-1]) for w in train_data_lines_right]

nbr_train = len(train_data_lines_right)
print('# Train Images of Binoculus: {}.'.format(nbr_train))

#将左右眼合并为一个tuple
train_data_lines = (train_data_lines_left, train_data_lines_right)

#一个epoch内的训练次数
steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))

#读入左眼测试集路径
val_path = os.path.join(LIST_ROOT, 'kaggle_val_left1.list')
val_data_lines_left = open(val_path).readlines()
val_data_lines_left = [w for w in val_data_lines_left if os.path.exists(w.strip().split(' ')[0])]
#读入右眼测试集路径
val_path = os.path.join(LIST_ROOT, 'kaggle_val_right1.list')
val_data_lines_right = open(val_path).readlines()
val_data_lines_right = [w for w in val_data_lines_right if os.path.exists(w.strip().split(' ')[0])]

nbr_val = len(val_data_lines_right)
print('# Val Images of Binoculus: {}.'.format(nbr_val))

val_data_lines = (val_data_lines_left, val_data_lines_right)

#一个epoch内的测试次数,当向fit_generator送入生成器时用到
validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))

#不平衡类别权重
if USE_CLASS_WEIGHTS:
    print('Using Class Balanced Weights ...')
    class_weights_left = class_weight.compute_class_weight('balanced', np.unique(train_labels_left), train_labels_left)
    class_weights_right = class_weight.compute_class_weight('balanced', np.unique(train_labels_right), train_labels_right)
    class_weights = [class_weights_left, class_weights_right]
    print('Class weight: {}.'.format(class_weights))
else:
    class_weights = None

#训练集batch生成器
train_generator = generator_img_batch(train_data_lines, nbr_classes = NBR_CLASSES, batch_size = BATCH_SIZE,
                                      img_width = IMG_WIDTH, img_height = IMG_HEIGHT, random_shuffle = True,
                                      preprocess = True, augment = True, crop = True, mirror = True)
#测试集batch生成器
validation_generator = generator_img_batch(val_data_lines, nbr_classes = NBR_CLASSES, batch_size = BATCH_SIZE,
                                           img_width = IMG_WIDTH, img_height = IMG_HEIGHT, random_shuffle = True,
                                           preprocess = True, augment = False, crop = True, mirror = False)

auc_curve = RocAucMetric(validation_generator, validation_steps, best_model_file, save_best=True)
early_stop = EarlyStopping(monitor='auc_val', patience=1, verbose=1)

#跑起来
print('Model training begins...')
model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                    callbacks = [auc_curve, early_stop ],
                    validation_data = validation_generator, validation_steps=validation_steps,
                    class_weight = class_weights, max_q_size = MAX_Q_SIZE, workers = WORKERS, pickle_safe=True)

# load the best model
best_model = load_model('./best_weight/siamese_inception.h5')
#plot auc curve
y_p = []
y_t = []

for i in range(validation_steps):
    batch = next(validation_generator)
    # val_data = [batch[0]['left_input'],batch[0]['right_input']]
    y_pred = best_model.predict(batch[0], batch_size=BATCH_SIZE, verbose=0)
    # print(type(y_pred[0]),'size:', y_pred[0].shape)
    pred_label = [np.squeeze(y_pred[0]), np.squeeze(y_pred[1])]
    y_p_batch = np.concatenate(pred_label)

    # 从字典中将左右眼的验证集标签取出
    val_label = [np.squeeze(batch[1]['left_output']), np.squeeze(batch[1]['right_output'])]
    y_t_batch = np.concatenate(val_label)
    # print(y_t_batch)
    # print(y_p_batch)
    # print(type(y_t_batch))
    # print(type(y_p_batch))
    # print(y_t_batch.shape)
    # print(y_p_batch.shape)

    y_p.append(y_p_batch)
    y_t.append(y_t_batch)

y_p = np.concatenate(y_p)
y_t = np.concatenate(y_t)
# acc = np.mean(np.equal(y_t, np.round(y_p)))
'''

y_t = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
y_p = [0.31689620142873609, 0.32367439192936548, 0.42600526758001989, 0.38769987193780364,
           0.3667541015524296, 0.39760831479768338, 0.42017521636505745, 0.41936155918127238,
           0.33803961944475219, 0.33998332945141224]
auc = roc_auc_score(y_t, y_p)
# loss = -np.mean(y_t*np.log(y_p+1e-10)+(1-y_t)*np.log(1-y_p+1e-10))
fpr, tpr, _ = roc_curve(y_t, y_p)

fix_sen = 0.980
fix_spc = 1.000 - 0.970
nearest_value_sen = np.unique(np.sort(np.square(tpr - fix_sen)))[0]
farer_value_sen = np.unique(np.sort(np.square(tpr - fix_sen)))[1]
high_value_sen = np.maximum(nearest_value_sen, farer_value_sen)
idx_high_sen = np.argwhere(np.square(tpr - fix_sen) == high_value_sen)[0]

nearest_value_spc = np.unique(np.sort(np.square(fpr - fix_spc)))[0]
farer_value_spc = np.unique(np.sort(np.square(fpr - fix_spc)))[1]
low_value_spc = np.minimum(nearest_value_spc, farer_value_spc)
idx_low_spc = np.argwhere(np.square(fpr - fix_spc) == low_value_spc)[0]

plt.figure()
plt.plot(fpr, tpr, color='black', label='ROC curve (area = %0.3f)' % auc)

plt.scatter(fpr[idx_high_sen], fix_sen, s=25, marker='o', color='red', label='High-sensitivity point',
            alpha=1.0)  # high sensitivity point
plt.vlines(fpr[idx_high_sen], 0, fix_sen, colors="r", linestyles="dashed", alpha=0.8, linewidth='0.8')
plt.hlines(fix_sen, 0, fpr[idx_high_sen], colors="r", linestyles="dashed", alpha=0.8, linewidth='0.8')
plt.text(fpr[idx_high_sen], fix_sen, (float('%.3f' % fpr[idx_high_sen]), float('%.3f' % fix_sen)),
         ha='left', va='top', fontsize=8)

plt.scatter(fix_spc, tpr[idx_low_spc], s=25, marker='o', color='g', label='High-specificity point'
            , alpha=1.0)  # high sensitivity point
plt.vlines(fix_spc, 0, tpr[idx_low_spc], colors="g", linestyles="dashed", alpha=0.8, linewidth='0.8')
plt.hlines(tpr[idx_low_spc], 0, fix_spc, colors="g", linestyles="dashed", alpha=0.8, linewidth='0.8')
plt.text(fix_spc, tpr[idx_low_spc], (float('%.3f' % fix_spc), float('%.3f' % tpr[idx_low_spc])),
         ha='left', va='top', fontsize=8)

plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
# plt.savefig('./roc_curve.pdf')
# plt.close()

# def classifaction_report_csv(y_true, y_pred):
#     row = {}
#     row['class'] = []
#     row['precision'] = []
#     row['recall'] = []
#     row['f1_score'] = []
#     row['support'] = []
#
#     acc = np.mean(np.equal(y_true, np.round(y_pred)))
#     report = classification_report(y_true, np.round(y_pred))
#     print(report)
#     lines = report.split('\n')
#     for i in [2, 3]:
#         line = lines[i]
#         row_data = line.split('      ')
#         row['class'].append(row_data[1])
#         row['precision'].append(float(row_data[2]))
#         row['recall'].append(float(row_data[3]))
#         row['f1_score'].append(float(row_data[4]))
#         row['support'].append(float(row_data[5]))
#
#     line = lines[5]
#     row_data = line.split('      ')
#     row['class'].append(row_data[0])
#     row['precision'].append(float(row_data[1]))
#     row['recall'].append(float(row_data[2]))
#     row['f1_score'].append(float(row_data[3]))
#     row['support'].append(float(row_data[4]))
#
#     row['accuracy'] = acc
#     dataframe = pd.DataFrame.from_dict(row)
#     dataframe.to_csv('./list/classification_report.csv', index=False)
#
#
# # save as csv file
# fpr_list = fpr.tolist()
# tpr_list = tpr.tolist()
# file = pd.DataFrame({'false_positive':fpr_list,'true positive':tpr_list})
# file.to_csv('./list/roc_data.csv')
# classifaction_report_csv(y_t, y_p)