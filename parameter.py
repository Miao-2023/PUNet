
def get_params():
    # ########### default parameters ##############
    params = dict(
        quick_test=True, # To do quick test. Trains/test on small subset of dataset, and # of epochs
        # quick_test=False, # To do quick test. Trains/test on small subset of dataset, and # of epochs
        nb_epochs= 500,  # 训练次数
        patient_times=10,  # ER没有改进的等待次数
        batch_size=1,  # Batch size 越小，迭代下降速度越快。越大，下降越准。
        fs=int(16e3), #采样率
        in_ch = 1,
        out_ch = 1,
        frames = 66,
        hop_len = 256,
        in_len = 512,

        # finetune_mode=False,
        finetune_mode=True,
        lr=(1e-3, 5e-4, 2.5e-4, 1e-4),
        lr_start=0,

        feat_train_dir = 'I:/VCTK2/train/noisy',
        label_train_dir = 'I:/VCTK2/train/clean',
        feat_val_dir = 'I:/VCTK2/test/noisy',
        label_val_dir = 'I:/VCTK2/test/clean',
        feat_test_dir = 'I:/VCTK2/test/noisy',
        label_test_dir = 'I:/VCTK2/test/clean',

        unique_name='PUNET2DVCTK1116',


        is_frame_shift=True,
        is_shuffle = True,
        model_dir='models/',
        dcase_dir='results/',
        log_dir= 'log/',
    )
    params['pretrained_model_weights']='{}{}_model.h5'.format(
        params['model_dir'],params['unique_name'])
    params['feat_shape'] = (params['batch_size'], params['in_ch'], params['frames'], params['in_len'])
    params['label_shape'] = (params['batch_size'], params['out_ch'], params['frames'], params['in_len'])
    # for key, value in params.items():
    #     print("{}: {}".format(key, value))
    return params
