import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import my_logging
import parameter as parameters
from MetricsSE import create_folder, SISDRmetrics, STOImetrics, PESQmetrics
from model.TorchOLA import TorchOLA
import DataGenPerFile as test_data_generator
from torch.utils.data.dataloader import DataLoader

from model.PUNET import PUNET as NetModel

def test_run(params, model, criterion, device, save_out=False, save_path=None):
    test = unseen_test(params=params, batch_size=1,
                       feat_path=params['feat_test_dir'],
                       label_path=params['label_test_dir'],
                       save_out = save_out, save_path = save_path)
    test.run_test(model=model, criterion=criterion, params=params, device=device)

class unseen_test():
    def __init__(self, params, feat_path, label_path,
                 split_name = None, split_dB = None, batch_size=1,
                 save_out=False, save_path = None):
        self._params = params
        self._split_name = split_name
        self._split_dB = split_dB
        self._feat_path = feat_path
        self._label_path = label_path
        self._batch_size = batch_size
        self._save_out = save_out
        self._save_path = save_path

    def run_test(self, model, criterion, params, device):
        data_gen = test_data_generator.DataGenerator(
            params=self._params, split_name=self._split_name, split_dB=self._split_dB,
            feat_path=self._feat_path, label_path=self._label_path)
        dataloader = DataLoader(dataset = data_gen, num_workers=0, batch_size= params['batch_size'])
        start_time = time.time()
        loss, sisdr, stoi, pesq = self.run_model(dataloader, model, criterion, params,device,
                                                 save_out=self._save_out, save_path=self._save_path)
        test_time = time.time() - start_time
        print('time: {:0.1f}, loss*1e4: {:0.1f}, sisdr: {:05.2f}, stoi%: {:0.2f}, pesq: {:0.2f}'.format(
            test_time, loss * 1e4, sisdr, stoi * 100., pesq)
        )

    # 验证函数
    def run_model(self, data_generator, model, criterion, params, device, save_out=False, save_path=None):
        my_sisdr = SISDRmetrics()
        my_stoi = STOImetrics(params['fs'])
        my_pesq = PESQmetrics(params['fs'])
        if save_out:
            create_folder(save_path)
        nb_test_batches, test_loss = 0, 0.
        model.eval()
        ola = TorchOLA(frame_shift=params['hop_len'])
        ham_window=torch.hamming_window(params['in_len']).cuda()
        file_list = data_generator.dataset.get_filelist()
        cnt_file = 0
        with torch.no_grad():
            for data, ref in data_generator:
                start_time = time.time()
                data_gpu = data.cuda().detach()
                ref_gpu = ref.cuda().detach()
                pred_wave = model(data_gpu)
                # ------------------OLA-------------------------------------
                data_ola = ola(data_gpu).squeeze()
                pred_ola = ola(pred_wave).squeeze()
                ref_ola = ola(ref_gpu).squeeze()
                loss = criterion(pred_ola, ref_ola, data_ola)
                # ------------------gpu to cpu-------------------------------------
                pred_out = pred_ola.squeeze().cpu().data.numpy()
                ref_out = ref_ola.squeeze().cpu().data.numpy()
                # ------------------SI-SDR-------------------------------------
                tmp_sisdr = my_sisdr.sisdr(pred_out, ref_out)
                my_sisdr.append(tmp_sisdr.cpu().data.numpy())
                # ------------------STOI-------------------------------------
                ## STOI、PESQ计算算法不支持GPU加速，转换成CPU计算.numpy
                tmp_stoi = my_stoi.stoi(pred_out, ref_out)
                my_stoi.append(tmp_stoi)
                # ------------------PESQ-------------------------------------
                tmp_pesq = my_pesq.pesq(pred_out, ref_out)
                if tmp_pesq>=0:
                    my_pesq.append(tmp_pesq)
                # pesq_time = time.time() - start_time
                # gpu2cpu_time = 0.043, stoi_time = 0.058, pesq_time = 0.126
                # print('gpu2cpu_time = {:0.3f}, stoi_time = {:0.3f}, pesq_time = {:0.3f}'.format(
                #         g2c_time, stoi_time,  pesq_time))
                # ------------------save-------------------------------------
                # file_name = data_generator.get_cnt_file()
                file_name = file_list[cnt_file]
                cnt_file += 1
                if save_out:  # 保存ref和pred的 ola_wave 数据
                    save_name = '{}/{}'.format(save_path,file_name)
                    save_mat = np.concatenate((pred_out.reshape(-1, 1), ref_out.reshape(-1, 1)), axis=1)
                    np.save(save_name, save_mat)

                test_loss += loss.item()
                nb_test_batches += 1
                if params['quick_test'] and nb_test_batches == 10:
                    break
                # last_time = time.time() - start_time
                # start_time = time.time()
                print('file name = {}, si-sdr[{:05.2f}], stoi%[{:05.2f}], pesq[{:04.2f}], time = {:0.3f}'.format(
                    file_name, tmp_sisdr, tmp_stoi * 100., tmp_pesq, time.time() - start_time))
            test_loss /= nb_test_batches

        out_sisdr = my_sisdr.output()
        my_sisdr.clear()
        out_stoi = my_stoi.output()
        my_stoi.clear()
        out_pesq = my_pesq.output()
        my_pesq.clear()
        return test_loss, out_sisdr, out_stoi, out_pesq


def main(argv):
    main_st_time = time.time()
    # use parameter set defined by user
    params = parameters.get_params()
    time_now = datetime.now()
    time_now = '{:2}{:02}{:02}_{:02}-{:02}-{:02}'.format(
        time_now.year % 100, time_now.month, time_now.day, time_now.hour, time_now.minute, time_now.second)
    unique_name = params['unique_name']
    create_folder(params['dcase_dir'])
    create_folder(params['log_dir'])
    dcase_output_log_folder = os.path.join(params['dcase_dir'], '{}_{}_log'.format(time_now, unique_name))

    # 记录日志
    if not params['quick_test']:
        sys.stdout = my_logging.Logger(dcase_output_log_folder)  # 开始日志记录
    print('argv = {}'.format(argv))
    for key, value in params.items():
        print("{}: {}".format(key, value))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)  # 在自动求导时检测错误产生路径

    print('Load model weights.')
    model = NetModel().to(device)
    model.load_state_dict(torch.load(params['pretrained_model_weights']), strict=False)
    criterion = nn.L1Loss()  # 用于计算两个输入对应元素的平均绝对误差（MAE）的均值

    # -------------------unseen test------------------------------
    test_run(params=params,model=model,criterion=criterion,device=device,
             save_out=True and not params['quick_test'],save_path='test_out')

    # -------------------Print program runtime and end time------------------------------
    main_cost_time = time.time() - main_st_time
    minute = int(main_cost_time / 60)
    hour = minute // 60
    minute = minute % 60
    print('Program runtime：{}h-{}m'.format(hour, minute))
    time_now = datetime.now()
    print('Program end time：{}:{}'.format(time_now.hour, time_now.minute))
    sys.stdout.flush()  # 写入日志文件并保存，避免中断丢失记录


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
