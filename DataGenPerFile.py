# 生成训练数据 包括输入特征和标签特征
# 特征和标签为时域数据
import math
import os
import numpy as np
from collections import deque
import random
from torch.utils.data.dataset import Dataset

class DataGenerator(Dataset):
    def __init__(
            self, params, feat_path, label_path, is_shuffle=False, iter_times=0,
            is_frame_shift=False, split_name = None, split_dB = None,
    ):
        self.split_name = split_name
        self.split_dB = split_dB
        self._fs = params['fs']
        self._in_len = params['in_len']
        self._hop_len = params['hop_len']
        self._frames = params['frames']
        self._feat = np.zeros((params['in_ch'], params['frames'], params['in_len']),dtype='float32')
        self._label = np.zeros((params['out_ch'], params['frames'], params['in_len']),dtype='float32')

        self._shuffle = is_shuffle
        self._is_frame_shift = is_frame_shift
        self._iter_times = iter_times
        self._feat_shape = self._feat.shape
        self._label_shape = self._label.shape
        self._feat_dir = feat_path
        self._label_dir = label_path
        self._batch_len = (self._frames-1)*self._hop_len + self._in_len

        #
        self._filenames_list = list()
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()
        self.cnt_file = 0
        self.cnt_file_name = None
        self.cnt_file_batch = 0
        self.cnt_file_batch_size = 0
        #
        self.flush_file()
        self._get_filenames_list_and_feat_label_sizes()

    def get_batch_len(self):
        return self._batch_len

    def flush_file(self):
        self.cnt_file = 0
        if self._shuffle:
            random.shuffle(self._filenames_list)

    def get_data_sizes(self):
        return self._feat_shape, self._label_shape

    def _get_filenames_list_and_feat_label_sizes(self):
        for filename in os.listdir(self._feat_dir):
            if not self.split_name and not self.split_dB:
                self._filenames_list.append(filename)
            elif self.split_name and self.split_dB:
                if self.split_name in filename and self.split_dB in filename:
                    self._filenames_list.append(filename)
            elif self.split_name:
                if self.split_name in filename:
                    self._filenames_list.append(filename)
            elif self.split_dB :
                if self.split_dB in filename:
                    self._filenames_list.append(filename)
        return 0

    def get_cnt_file_inf(self):
        return self.cnt_file_name, self.cnt_file_batch, self.cnt_file_batch_size

    def get_filelist(self):
        return self._filenames_list

    def __len__(self):
        if self._iter_times >0:
            return self._iter_times
        else:
            self._iter_times = len(self._filenames_list)
            print("iter_times = {}".format(self._iter_times))
            return self._iter_times

    def __getitem__(self, idx):
        if self.cnt_file >= len(self._filenames_list):
            self.flush_file()
            print('__getitem__：迭代文件超出长度，执行 self.flush_file()')
        file_name = self._filenames_list[self.cnt_file]
        self.cnt_file = self.cnt_file + 1

        feat_file = np.load(os.path.join(self._feat_dir, file_name)).astype(np.float32).reshape(-1)
        label_file = np.load(os.path.join(self._label_dir, file_name)).astype(np.float32).reshape(-1)

        if len(feat_file) != len(label_file):
            print('ERROR：feat file 和 label file 长度不一致：{}'.format(file_name))

        frame_num = math.ceil( (len(feat_file) - self._in_len) / self._hop_len) + 1
        data_len = (frame_num-1) * self._hop_len + self._in_len
        feat_file = np.concatenate((feat_file,np.zeros(data_len-len(feat_file))))
        label_file = np.concatenate((label_file,np.zeros(data_len-len(label_file))))
        feat=np.zeros((1,frame_num,self._in_len),dtype='float32')
        label=np.zeros((1,frame_num,self._in_len),dtype='float32')
        for t in range(frame_num):
            psn = t * self._hop_len
            feat[0,t,:] = feat_file[psn:psn + self._in_len]
            label[0,t,:] = label_file[psn:psn + self._in_len]
        return feat, label
