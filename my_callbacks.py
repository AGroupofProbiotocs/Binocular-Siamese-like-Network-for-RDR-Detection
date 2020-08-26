# -*- coding: utf-8 -*-
"""
========================================================================
A siamese-like CNN for diabetic retinopathy detection using binocular 
fudus images as input, Version 1.0
Copyright(c) 2020 Xianglong Zeng, Haiquan Chen, Yuan Luo, Wenbin Ye
All Rights Reserved.
----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is here
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
----------------------------------------------------------------------
Please cite the following paper when you use it:
X. Zeng, H. Chen, Y. Luo and W. Ye, "Automated Diabetic Retinopathy 
Detection Based on Binocular Siamese-Like Convolutional Neural 
Network," in IEEE Access, vol. 7, pp. 30744-30753, 2019, 
doi: 10.1109/ACCESS.2019.2903171.
----------------------------------------------------------------------
This file defines some callbacks.

@author: Xianglong Zeng
========================================================================
"""
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import numpy as np
import warnings
from keras import backend as K

# recording loss history
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = []
        self.val_acc = []
        self.count = 0

    def on_batch_end(self, batch, logs={}):
        self.count += 1
        if self.count%50 == 0:
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.count = 0
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

    def vis_losses(self, typ='epoch'):
        # 画出损失曲线
        plt.figure()
        plt.plot(np.arange(len(self.losses[typ])), self.losses[typ], 'r', label='train_losses')
        plt.plot(np.arange(len(self.val_loss)), self.val_loss, 'b', label='val_losses')
        try:
            plt.plot(np.arange(len(self.accuracy[typ])), self.accuracy[typ][0], 'g', label='train_acc')
            plt.plot(np.arange(len(self.val_acc)), self.val_acc, 'k', label='val_acc')
        except:
            print(self.val_loss)
        plt.xlabel(typ)
        plt.ylabel('train-accuracy')
        plt.legend()
        plt.title('The loss curve during training process')
        # plt.show()
        plt.savefig('./loss_curve.pdf')
        plt.close()

    def on_train_end(self, logs=None):
        self.vis_losses()

        
class RocAucMetric(Callback):  
    def __init__(self, validation_generator, validation_steps, filepath, save_best=False, early_stop=False, patience=0):
        super(RocAucMetric, self).__init__()
        self.auc = None
        self.best = - np.inf
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.save_best = save_best
        self.filepath = filepath
        self.patience = patience
        self.early_stop = early_stop
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        if not ('auc_val' in self.params['metrics']):
            self.params['metrics'].append('auc_val')
        logs['auc_val'] = - np.inf
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        y_p = []
        y_t = []

        for i in range(self.validation_steps):
            batch = next(self.validation_generator)
            y_pred = self.model.predict(batch[0], batch_size=len(batch[0]), verbose=0)
            pred_label = [np.squeeze(y_pred[0]), np.squeeze(y_pred[1])]
            y_p_batch = np.concatenate(pred_label)

            # 从字典中将左右眼的验证集标签取出
            val_label = [np.squeeze(batch[1]['left_output']), np.squeeze(batch[1]['right_output'])]
            y_t_batch = np.concatenate(val_label)

            y_p.append(y_p_batch)
            y_t.append(y_t_batch)

        y_p = np.concatenate(y_p)
        y_t = np.concatenate(y_t)
        self.auc = roc_auc_score(y_t, y_p)

        logs['auc_val'] = self.auc

        current = self.auc
        if current > self.best:
            if self.save_best:
                print('\nEpoch %05d: auc_val improved from %0.5f to %0.5f,' ' saving model to %s'
                      % (epoch + 1, self.best, current, self.filepath))
                self.model.save(self.filepath, overwrite=True)
            if self.early_stop:
                self.wait = 0
            self.best = current

        else:
            if self.save_best:
                print('\nEpoch %05d: auc_val did not improve' %(epoch + 1))
            if self.early_stop:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.early_stop and self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))



        # print('test',logs.get('auc_val'))

            

class LRWithWarmRestart(Callback):
    def __init__(self, monitor='val_loss', factor=0.1, patience=10, min_lr=0, auto = True):
        super(LRWithWarmRestart, self).__init__()
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.wait = 0
        # self.lr_epsilon = self.min_lr * 1e-4
        self.origin_lr = 0.001
        self.best = 0
        self.monitor_op = None
        self.auto = auto

    def on_train_begin(self, logs=None):
        self.wait = 0
        # self.lr_epsilon = self.min_lr * 1e-4
        self.origin_lr = K.get_value(self.model.optimizer.lr)
        self.monitor_op = lambda a, b: np.less(a, b)
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.auto:
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                else:
                    if self.factor > 1:
                        self.factor = 1
                    else:
                        self.factor *= 1.3
                    self.origin_lr *= 0.75
                    new_lr = self.origin_lr  # warm restart
                K.set_value(self.model.optimizer.lr, new_lr)
                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch + 1, new_lr))
                self.wait = 0
            self.wait += 1