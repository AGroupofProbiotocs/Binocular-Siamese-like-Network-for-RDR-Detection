from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import Model, load_model
from keras import backend as K
from batch_generator import generator_img_batch
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

DROPOUT_KEEP_PROB = 0.5
LIST_ROOT = './list/'
LEARNING_RATE = 0.001
NBR_EPOCHS = 200
BATCH_SIZE = 32
IMG_WIDTH = 299
IMG_HEIGHT = 299
MONITOR_INDEX = 'loss'
NBR_CLASSES = 1
USE_CLASS_WEIGHTS = False
GPUS = "2"
MAX_Q_SIZE = 100
WORKERS = 16
FINE_TUNE = False

os.environ["CUDA_VISIBLE_DEVICES"] = GPUS

test_path = os.path.join(LIST_ROOT, 'kaggle_test_left.list')
test_data_lines_left = open(test_path).readlines()
test_data_lines_left = [w for w in test_data_lines_left if os.path.exists(w.strip().split(' ')[0])]
#读入右眼测试集路径
test_path = os.path.join(LIST_ROOT, 'kaggle_test_right.list')
test_data_lines_right = open(test_path).readlines()
test_data_lines_right = [w for w in test_data_lines_right if os.path.exists(w.strip().split(' ')[0])]

nbr_test = len(test_data_lines_right)

test_data_lines = (test_data_lines_left, test_data_lines_right)

#一个epoch内的测试次数,当向fit_generator送入生成器时用到
test_steps = int(np.ceil(nbr_test* 1. / BATCH_SIZE))

test_generator = generator_img_batch(test_data_lines, nbr_classes = NBR_CLASSES, batch_size = BATCH_SIZE,
                                           img_width = IMG_WIDTH, img_height = IMG_HEIGHT, random_shuffle = True,
                                           preprocess = True, augment = False, crop = True, mirror = False)

best_model = load_model('./best_weight/auc_with_512dense&dropout&mirrored_preprocessing&adam&auc.h5')

#plot auc curve
y_p = []
y_t = []

for i in range(test_steps):
    batch = next(test_generator)
    # val_data = [batch[0]['left_input'],batch[0]['right_input']]
    y_pred = best_model.predict(batch[0], batch_size=BATCH_SIZE, verbose=0)
    # print(type(y_pred[0]),'size:', y_pred[0].shape)
    pred_label = [np.squeeze(y_pred[0]), np.squeeze(y_pred[1])]
    y_p_batch = np.concatenate(pred_label)

    # 从字典中将左右眼的验证集标签取出
    val_label = [np.squeeze(batch[1]['left_output']), np.squeeze(batch[1]['right_output'])]
    y_t_batch = np.concatenate(val_label)

    y_p.append(y_p_batch)
    y_t.append(y_t_batch)

y_p = np.concatenate(y_p)
y_t = np.concatenate(y_t)
# acc = np.mean(np.equal(y_t, np.round(y_p)))
auc = roc_auc_score(y_t, y_p)
# loss = -np.mean(y_t*np.log(y_p+1e-10)+(1-y_t)*np.log(1-y_p+1e-10))
fpr, tpr, _ = roc_curve(y_t, y_p)

fix_sen = 0.950
fix_spc = 1.000 - 0.950
idx_high_sen = np.argwhere(np.diff(np.sign(tpr - fix_sen)) != 0).reshape(-1)[0] + 1
idx_low_spc = np.argwhere(np.diff(np.sign(fpr - fix_spc)) != 0).reshape(-1)[0] + 0

refer_x = np.linspace(0, 1, len(fpr))
refer_y = refer_x

plt.figure()
plt.plot(fpr, tpr, color='black', label='ROC curve (area = %0.3f)' % auc)
plt.plot(refer_x, refer_y, color='gray', label='Reference line', ls="dashed", alpha=0.5)

plt.scatter(fpr[idx_high_sen], fix_sen, s=20, marker='o', color='red', label='High-sensitivity point',
            alpha=1.0)  # high sensitivity point
plt.vlines(fpr[idx_high_sen], 0, fix_sen, colors="r", linestyles="dashed", alpha=0.8, linewidth=0.8)
plt.hlines(fix_sen, 0, fpr[idx_high_sen], colors="r", linestyles="dashed", alpha=0.8, linewidth=0.8)
plt.text(fpr[idx_high_sen], fix_sen, (float('%.3f' % fpr[idx_high_sen]), float('%.3f' % fix_sen)),
         ha='left', va='top', fontsize=10)

plt.scatter(fix_spc, tpr[idx_low_spc], s=20, marker='o', color='g', label='High-specificity point'
            , alpha=1.0)  # high sensitivity point
plt.vlines(fix_spc, 0, tpr[idx_low_spc], colors="g", linestyles="dashed", alpha=0.8, linewidth=0.8)
plt.hlines(tpr[idx_low_spc], 0, fix_spc, colors="g", linestyles="dashed", alpha=0.8, linewidth=0.8)
plt.text(fix_spc, tpr[idx_low_spc], (float('%.3f' % fix_spc), float('%.3f' % tpr[idx_low_spc])),
         ha='left', va='top', fontsize=10)

plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig('./images/roc_curve_test.pdf')
plt.close()
