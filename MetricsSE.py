import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plot
import math
import os
import time
from pesq import pesq
from pystoi.stoi import stoi
plot.switch_backend('agg')#加上这一句以后不显示图像

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

class SE_loss_plot:
    def __init__(self):
        self.train_para = np.array([])
        self.eval_para = np.array([])

    def record(self,train_loss, eval_loss):
        self.train_para = np.append(self.train_para, [train_loss])
        self.eval_para = np.append(self.eval_para, [eval_loss])

    def plot(self,fig_name,is_save=False):
        if self.train_para.size == 0 or self.eval_para.size == 0:
            return
        train_para = self.train_para.reshape(-1)
        eval_para = self.eval_para.reshape(-1)
        # best_loss = eval_para[:,0].argmin()
        best_loss = eval_para.argmin()

        plot.figure(figsize=(12.8,7.2))
        plot.plot(range(1, 1 + train_para.shape[0]), train_para[:], label='train loss')
        plot.plot(range(1, 1 + eval_para.shape[0]), eval_para[:], label='val loss')
        plot.legend()
        plot.grid(True)
        plot.title('best loss [ train/eval (1e4): {:0.2f} / {:0.2f} ]'.format(
            train_para[best_loss]*1e4,eval_para[best_loss]*1e4))

        if is_save:
            save_path='{}.png'.format(fig_name)
            plot.savefig(save_path)
            plot.close()

class SISDRmetrics():
    def __init__(self):
        self.record = np.array([])

    def sisdr(self, pred_wave, ref_wave):#tensor，支持gpu运算
        if not torch.is_tensor(pred_wave):
            pred_wave = torch.as_tensor(pred_wave)
        if not torch.is_tensor(ref_wave):
            ref_wave = torch.as_tensor(ref_wave)
        S_t = torch.sum(pred_wave * ref_wave) * ref_wave / ref_wave.square().sum()
        S_e = pred_wave - S_t
        sisdr = 10*torch.log10(S_t.square().sum() / S_e.square().sum()) #[-inf,inf]
        return sisdr

    def sisdr_cpu(self, pred_wave, ref_wave):#numpy转tensor运算
        pred_wave = torch.as_tensor(pred_wave)
        ref_wave = torch.as_tensor(ref_wave)
        if not torch.is_tensor(pred_wave):
            pred_wave = torch.as_tensor(pred_wave)
        if not torch.is_tensor(ref_wave):
            ref_wave = torch.as_tensor(ref_wave)
        S_t = torch.sum(pred_wave * ref_wave) * ref_wave / ref_wave.square().sum()
        S_e = pred_wave - S_t
        sisdr = 10*torch.log10(S_t.square().sum() / S_e.square().sum())
        return sisdr

    def append(self,input):
        self.record = np.append(self.record, input)

    def clear(self):
        self.record = np.array([])

    def output(self):
        if len(self.record):
            return self.record.mean()
        else:
            return -float('inf')

class STOImetrics():
    def __init__(self, fs):
        self.record = np.array([])
        self.fs = fs

    def stoi(self,pred_wave, ref_wave):
        try:
            if len(pred_wave.shape) == 2:
                out_stoi = 0
                for b in range(pred_wave.shape[0]):
                    tmp_stoi = stoi(ref_wave[b,:], pred_wave[b,:], self.fs)
                    out_stoi = out_stoi + tmp_stoi
                out_stoi = out_stoi/pred_wave.shape[0]
            elif len(pred_wave.shape) == 1:
                out_stoi = stoi(ref_wave, pred_wave, self.fs)
            else:
                raise Exception("error stoi shape")
        except:
            raise Exception("error stoi")
        return out_stoi

    def append(self,input):
        self.record = np.append(self.record, input)

    def clear(self):
        self.record = np.array([])

    def output(self):
        if len(self.record):
            return self.record.mean()
        else:
            return -float('inf')

class PESQmetrics():
    def __init__(self, fs):
        self.record = np.array([])
        self.fs = fs

    def pesq(self, pred_wave, ref_wave):
        try:
            out_pesq = pesq(self.fs, ref_wave, pred_wave, 'wb')
        except:
            raise Exception("error pesq")
        return out_pesq

    def append(self,input):
        self.record = np.append(self.record, input)

    def clear(self):
        self.record = np.array([])

    def output(self):
        if len(self.record):
            return self.record.mean()
        else:
            return -float('inf')

class Loss_LT(nn.Module):
    def __init__(self):
        super(Loss_LT, self).__init__()

    def forward(self, pred_wave, ref_wave):
        loss_lt = (pred_wave - ref_wave).abs().mean()
        return loss_lt


class Loss_SM(nn.Module):
    def __init__(self):
        super(Loss_SM, self).__init__()

    def forward(self, pred_wave, ref_wave):
        pred_STFT = torch.fft.rfft(pred_wave)
        pred_stft_real = pred_STFT.real.abs()
        pred_stft_imag = pred_STFT.imag.abs()
        ref_STFT = torch.fft.rfft(ref_wave)
        ref_stft_real = ref_STFT.real.abs()
        ref_stft_imag = ref_STFT.imag.abs()
        loss1 = (pred_stft_real -  ref_stft_real).abs().mean()
        loss2 = (pred_stft_imag - ref_stft_imag).abs().mean()
        loss = (loss1 + loss2)/2
        return loss

class Loss_PCM(nn.Module):
    def __init__(self):
        super(Loss_PCM, self).__init__()
        self.loss_sm1 = Loss_SM()
        self.loss_sm2 = Loss_SM()

    def forward(self, pred_wave, ref_wave, mixture):
        loss_sm1 = self.loss_sm1(pred_wave, ref_wave)
        pred_nis = mixture - pred_wave
        ref_nis = mixture - ref_wave
        loss_sm2 = self.loss_sm2(pred_nis, ref_nis)
        loss_pcm = (loss_sm1 + loss_sm2)/2
        return loss_pcm


class Loss_LT_LPCM(nn.Module):
    def __init__(self):
        super(Loss_LT_LPCM, self).__init__()
        self.loss_lt = Loss_LT()
        self.loss_lpcm = Loss_PCM()
        self.alpha = 0.95

    def forward(self, pred_wave, ref_wave, mixture):
        loss_lt = self.loss_lt(pred_wave, ref_wave)
        loss_lpcm = self.loss_lpcm(pred_wave, ref_wave, mixture)
        loss = self.alpha * loss_lt + (1-self.alpha) * loss_lpcm
        return loss