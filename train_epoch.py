import os
import sys
import DataGenPerFile as data_generator
from torch.utils.data.dataloader import DataLoader
from MetricsSE import SE_loss_plot, create_folder, STOImetrics, Loss_LT_LPCM
import time
from datetime import datetime
import torch
import torch.optim as optim
import my_logging
import parameter as parameters
from model.TorchOLA import TorchOLA
from test_epoch import test_run

from model.PUNET import PUNET as NetModel

def train_epoch(data_generator, optimizer, model, criterion, params):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    data_generator.dataset.flush_file()
    ola = TorchOLA(frame_shift=params['hop_len'])
    # start_time = time.time()
    for data, ref in data_generator:
        data_gpu = data.cuda().detach()
        ref_gpu = ref.cuda().detach()
        optimizer.zero_grad()
        pred_wave = model(data_gpu)
        # ------------------OLA-------------------------------------
        data_ola = ola(data_gpu).squeeze()
        pred_ola = ola(pred_wave).squeeze()
        ref_ola = ola(ref_gpu).squeeze()
        loss = criterion(pred_ola, ref_ola, data_ola)
        # train_time = time.time() - start_time
        # start_time = time.time()
        loss.backward()
        optimizer.step()
        # backward_time = time.time() - start_time
        # start_time = time.time()

        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 10:
            break
        # print('load_data_time = {:0.3f}, train_time = {:0.3f}, backward_time = {:0.3f}'.format(
        #     load_data_time, train_time, backward_time))
        # start_time = time.time()

    train_loss /= nb_train_batches
    return train_loss

#验证函数
def val_epoch(data_generator, model, criterion, params):
    my_stoi = STOImetrics(params['fs'])
    nb_val_batches, val_loss = 0, 0.
    model.eval()
    data_generator.dataset.flush_file()#主要作用是把循环位置零和打乱文件顺序
    ola = TorchOLA(frame_shift=params['hop_len'])
    with torch.no_grad():
        # start_time = time.time()
        for data, ref in data_generator:
            # start_time = time.time()
            data_gpu = data.cuda().detach()
            ref_gpu = ref.cuda().detach()
            pred_wave = model(data_gpu)
            # ------------------OLA-------------------------------------
            data_ola = ola(data_gpu).squeeze()
            pred_ola = ola(pred_wave).squeeze()
            ref_ola = ola(ref_gpu).squeeze()
            loss = criterion(pred_ola, ref_ola, data_ola)
            # ------------------GPU to CPU-------------------------------------
            pred_out = pred_ola.cpu().data.numpy()
            ref_out = ref_ola.cpu().data.numpy()
            # ------------------STOI-------------------------------------
            tmp_stoi = my_stoi.stoi(pred_out, ref_out)
            my_stoi.append(tmp_stoi)
            # val_time = time.time() - start_time
            # start_time = time.time()
            val_loss += loss.item()
            nb_val_batches += 1
            if params['quick_test'] and nb_val_batches == 10:
                break
            # last_time = time.time() - start_time
            # start_time = time.time()
            # print('load_data_time = {:0.3f}, val_time = {:0.3f}, last_time = {:0.3f}'.format(
            #     load_data_time, val_time,  last_time))

        val_loss /= nb_val_batches
    out_stoi = my_stoi.output()
    my_stoi.clear()
    return val_loss,out_stoi

def main(argv):
    main_st_time=time.time()
    # use parameter set defined by user
    params = parameters.get_params()
    unique_name = params['unique_name']
    time_now = datetime.now()
    time_now = '{:2}{:02}{:02}_{:02}-{:02}-{:02}'.format(
        time_now.year%100, time_now.month, time_now.day, time_now.hour, time_now.minute, time_now.second)
    # Unique name for the run
    create_folder(params['model_dir'])
    create_folder(params['dcase_dir'])
    create_folder(params['log_dir'])
    #记录日志
    dcase_output_train_folder = os.path.join(params['log_dir'], '{}_{}_train'.format(time_now, unique_name))
    if not params['quick_test']:
        sys.stdout = my_logging.Logger(dcase_output_train_folder)
    print('argv = {}'.format(argv))
    for key, value in params.items():
        print("{}: {}".format(key, value))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)
    print('Loading train dataset.')
    data_gen_train = data_generator.DataGenerator(
        params=params, feat_path=params['feat_train_dir'], label_path=params['label_train_dir'],
        is_frame_shift=params['is_frame_shift'], is_shuffle=params['is_shuffle'])
    train_dataloader = DataLoader(dataset=data_gen_train, num_workers=0,batch_size= params['batch_size'])
    print('Loading validate dataset.')
    data_gen_val = data_generator.DataGenerator(
        params=params, feat_path=params['feat_val_dir'], label_path=params['label_val_dir'])
    val_dataloader = DataLoader(dataset = data_gen_val, num_workers=0, batch_size= params['batch_size'])
    model = NetModel().to(device)
    lr_cnt = params['lr_start']
    lr = params['lr'][lr_cnt]
    print('cnt_lr = {}'.format(lr))
    if params['finetune_mode']:
        print('Run in fine-tuning mode, initialize the model as weights - {}'.format(
            params['pretrained_model_weights']))
        model.load_state_dict(torch.load(params['pretrained_model_weights']),strict=False)

    print('---------------- SE-net -------------------')
    data_in, data_out = params['feat_shape'], params['label_shape']
    print('FEATURES:\tdata_in: {}\tdata_out: {}'.format(data_in, data_out))
    # start training
    record_name = os.path.join(params['model_dir'], '{}_{}'.format(time_now, unique_name))
    best_val_loss, best_val_epoch, best_val_metrics = float('inf'), 0, -float('inf'),
    plot_loss = SE_loss_plot()

    patient_times=0
    nb_epoch = 10 if params['quick_test'] else params['nb_epochs']
    optimizer = optim.Adam(model.parameters(), lr= lr, weight_decay=0)
    criterion = Loss_LT_LPCM() #MAE loss
    # criterion = Loss_SM1118() #MAE loss

    sys.stdout.flush()#写入日志文件并保存，避免中断丢失记录
    for epoch_cnt in range(nb_epoch):
        # ------------------TRAINING -------------------------------------
        start_time = time.time()
        train_loss  = train_epoch(train_dataloader, optimizer, model, criterion, params)
        train_time = time.time() - start_time
        # -------------------VALIDATION ------------------------------
        start_time = time.time()
        val_loss,val_metrics = val_epoch(val_dataloader, model, criterion, params)
        val_time = time.time() - start_time
        # -------------------records and print------------------------------
        if best_val_metrics < val_metrics :
            best_val_epoch = epoch_cnt
            # best_val_loss = val_loss
            best_val_metrics = val_metrics
            if not params['quick_test']:
                torch.save(model.state_dict(), params['pretrained_model_weights'])
            patient_times=0
        print('e/be[{}/{}], STOI%:[{:06.3f}], train/val loss*1e4 [{:0.1f}/{:0.1f}], time: {:0.1f}/{:0.1f}'.format(
            epoch_cnt+1, best_val_epoch+1, val_metrics*1e2, train_loss*1e4, val_loss*1e4, train_time, val_time))
        if epoch_cnt > 1 and epoch_cnt != best_val_epoch:
            patient_times+=1
            if patient_times % params['patient_times'] == 0:
                model.load_state_dict(torch.load(params['pretrained_model_weights']))
                lr_cnt += 1
                if lr_cnt >= len(params['lr']):
                    break
                lr = params['lr'][lr_cnt]
                optimizer.param_groups[0]['lr'] = lr
                print('The learning rate has decreased to：{}*1e5'.format(int(lr*1e5)))
                plot_loss.plot(record_name, is_save = not params['quick_test'])
        sys.stdout.flush()
        plot_loss.record(train_loss, val_loss)
    plot_loss.plot(record_name,is_save= not params['quick_test'])

    # -------------------UNSEEN ------------------------------
    print('Loading best model weights.')
    model.load_state_dict(torch.load(params['pretrained_model_weights']))
    # print('\nLoading val dataset.\n')
    # val_run(params=params,model=model,criterion=criterion,device=device)
    print('\nLoading test dataset.\n')
    test_run(params=params,model=model,criterion=criterion,device=device)
    # -------------------Print program runtime and end time------------------------------
    main_cost_time=time.time()-main_st_time
    minute = int(main_cost_time/60)
    hour = minute//60
    minute = minute % 60
    print('Program runtime：{}h-{}m'.format(hour, minute))
    time_now = datetime.now()
    print('Program end time：{}:{}'.format(time_now.hour,time_now.minute))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
