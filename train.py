"""Main KWS CNN training script"""

import argparse
import os
import shutil
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torchspeech.utils.data as td
from torchspeech.utils.modelcheckpoint import ModelCheckPoint
import threading
# Aux scripts
import kwspipe
import kwscnn
import modelconfig
from advertorch.attacks import L2PGDAttack


def main():
    """Main program entry"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--save_path', default='./rnncnn_models_beta_lessnoise')
    parser.add_argument('--print_tick', type=int, default=500)
    parser.add_argument('--save_tick', type=int, default=10000)
    parser.add_argument('--local', type=bool, default=False)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--mode', default='word')
    parser.add_argument('--datablob', required=True)
    parser.add_argument('--train_epochs', type=int, default=400)
    parser.add_argument('--file_set', required=True)

    parser.add_argument('--noise_type',
                        type=str,
                        required=True,
                        choices=['noise_batch', 'additive_rir_noise', 'none'])
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument("--gpu", type=str, default='0', help="gpu")
    parser.add_argument("--is_training",
                        type=bool,
                        default=True,
                        help="True for training")

    args = parser.parse_args()
    pin_mem = True

    # Prep the models
    gpu_str = str()
    for gpu in args.gpu.split(','):
        gpu_str = gpu_str + str(gpu) + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    # device = torch.device('cuda:{}'.format(args.gpu))
    # torch.cuda.set_device(args.gpu)

    models_config = [
        modelconfig.RC03(),
        # modelconfig.RC04(),
        # modelconfig.RC05()
    ]

    model_names = []
    model_paths = []
    log_paths = []
    models = []
    tensorboards = []
    LOGS = []

    for i, config in enumerate(models_config):
        # device = devices#[i % len(devices)]
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        MODEL_NAME = 'RNNCNN' + '-LR{}-rank{}-channel{}-epsf-{}'.format(0.001, config.rank, config.cnn_channels, config.eps_factor)
        if args.noise_type == 'noise_batch':
            MODEL_NAME += '-NoiseBatch'
        elif args.noise_type == 'additive_rir_noise':
            MODEL_NAME += '-NoiseRirSynthesis'

        MODEL_PATH = args.save_path + '/' + MODEL_NAME

        model_path = MODEL_PATH + '/checkpoints'
        log_path = MODEL_PATH + '/logs'

        model_names.append(MODEL_NAME)
        model_paths.append(model_path)
        log_paths.append(log_path)

        model = kwscnn.DSCNN_RNN_Block(config.cnn_channels,
                                config.rnn_hidden_size,
                                1,
                                device,
                                rank=config.rank, fwd_context=15)

        model = {'name': model.__name__, 'data': model}
        model['epsf'] = config.eps_factor
        model['data'] = model['data'].to(device)
        model['data'] = torch.nn.DataParallel(model['data'])
        if args.optim == "adam":
            model['opt'] = torch.optim.Adam(model['data'].parameters(),
                                            lr=config.lr)
            # model['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     model['opt'], T_max=args.train_epochs, eta_min=10e-6)
        #Adding support for SGD optimizer with LR scheduler as Cosine
        if args.optim == "sgd":
            model['opt'] = torch.optim.SGD(model['data'].parameters(),
                                           lr=config.lr)
            # model['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     model['opt'], T_max=args.train_epochs, eta_min=10e-6)

        
        # Initialize the logging
        LOG = logging.getLogger(MODEL_NAME)
        os.makedirs(log_path, exist_ok=True)
        logfile_format = logging.Formatter(
            '%(asctime)s\t%(name)s\t%(levelname)s\tP%(process)d\t%(message)s')
        root_logger = logging.getLogger(MODEL_NAME)
        root_logger.setLevel(logging.INFO)
        filehandler = logging.handlers.RotatingFileHandler(
            os.path.join(log_path, 'log.txt'),
            maxBytes=1024 * 1024 * 5,
            backupCount=100)
        filehandler.setFormatter(logfile_format)
        root_logger.addHandler(logging.StreamHandler())
        root_logger.addHandler(filehandler)
        tensorboard = SummaryWriter(log_path)
        LOG.info('Initialized logging')

        model['stats'] = None
        name = model['name']
        model['frame_rate'] = 3  # default frame rate is 3x (30ms)
        
        bootstraps = {}
        bootstrap = bootstraps[name] if name in bootstraps else None
        model['checkpoint'] = ModelCheckPoint(model['name'], model_path,
                                              model['data'], model['opt'],
                                              bootstrap, LOG=LOG)
        model['device'] = device

        models.append(model)

        os.makedirs(model_path, exist_ok=True)
        shutil.copy(__file__,
                    os.path.join(model_path, os.path.basename(__file__)))
        shutil.copy(
            kwscnn.__file__,
            os.path.join(model_path, os.path.basename(kwscnn.__file__)))
        shutil.copy(
            kwspipe.__file__,
            os.path.join(model_path, os.path.basename(kwspipe.__file__)))



        LOG.info(args)
        LOG.info(config)    
        # import pdb;pdb.set_trace()
        i16_size = sum([param.numel() for param in model['data'].parameters()
                        ]) / 1024.0 / 1024.0
        LOG.info('Model %s:  I8 size = %f', model['name'], i16_size)

        tensorboards.append(tensorboard)
        LOGS.append(LOG)

    if args.noise_type == 'noise_batch':
        data_pipe = kwspipe.make_noise_phoneme_pipe(local=args.local,
                                                    args=args)
    elif args.noise_type == 'additive_rir_noise':
        data_pipe = kwspipe.make_noise_rir_synthesis_phoneme_pipe_train(
            local=args.local, args=args)
    else:
        data_pipe = kwspipe.make_phoneme_pipe(local=args.local, args=args)

    pipe = td.DataPipeLoader(dataset=data_pipe,
                             pin_memory=pin_mem,
                             collate_fn=_noop,
                             num_workers=args.workers)
    # print(type(pipe))
    # print(pipe)

    threads = []

    for count, item in enumerate(pipe):

        # print(f'item is {item}')
        feature = item['feature']
        f_shape = feature.shape
        label = item['label']
        seqlen = item['seqlen']
            
        for t in threads:
            t.join()

        threads = []

        for i in range(len(models)):
            device = models[i]['device']

            eps_factor = model['epsf']            
            ep_radius = eps_factor * (1/224) * np.sqrt(f_shape[1] * f_shape[2]) * 20
            adversary = L2PGDAttack(
            models[i]['data'], loss_fn=kwscnn.loss_function_combined(), eps=ep_radius,
            nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=20.0,
            targeted=False)

            t = threading.Thread(target=_process_phoneme_model,
                                 args=(models[i], feature.to(device),
                                       label.to(device),
                                       count, args, tensorboards[i],
                                       LOGS[i], True, device, adversary))
            t.start()
            threads.append(t)

    for i in range(len(models)):
        tensorboards[i].close()
    print('done')


def _noop(value):
    return value[0]

def _process_phoneme_model(model, feature, label, count, args, tensorboard, LOG,
                           isfirst, device, adversary):
    # print(f'feature shape {feature.shape}')
    # print(f'label shape {label.shape}')
    # print('--------------------------')

    model_data = model['data']
    opt = model['opt']
    checkpoint = model['checkpoint']
    frame_rate = model['frame_rate']
    pad = 5  # ENSURE +VE

    #DataAugmentation for bricking
    # import pdb;pdb.set_trace()
    div_factor = 3
    mod_len = feature.shape[1]
    pad_len_mod = (div_factor - mod_len % div_factor) % div_factor
    pad_len_feature = pad_len_mod
    pad_data = torch.zeros(feature.shape[0], pad_len_feature,
                           feature.shape[2]).to(device)
    feature = torch.cat((feature, pad_data), dim=1)

    assert (feature.shape[1]) % div_factor == 0
    #Augmenting the label accordingly
    pad_len_label = pad_len_feature
    pad_data = torch.ones(label.shape[0], pad_len_label).to(device) * (-100)
    pad_data = pad_data.type(torch.long)
    label = torch.cat((label, pad_data), dim=1)

    step = checkpoint.global_step()
    # print(feature)
    features = feature.permute((0, 2, 1))  # NLC to NCL
    
    adv_untargeted = adversary.perturb(features, label)
    features = torch.cat((features, adv_untargeted), 0)
    label = torch.cat((label, label), 0)
    posteriors = model_data(features)
    N, C, L = posteriors.shape

    trim_label = label[:, ::frame_rate]  # 30ms tick
    # trim_label = trim_label[:, -(L + pad):-pad]
    trim_label = trim_label[:, :L]

    flat_posteriors = posteriors.permute((0, 2, 1))  # TO NLC
    flat_posteriors = flat_posteriors.reshape((-1, C))  # to [NL] x C
    flat_labels = trim_label.reshape((-1))

    _, idx = torch.max(flat_posteriors, dim=1)
    correct_count = (idx == flat_labels).sum().detach()
    valid_count = (flat_labels >= 0).sum().detach()

    opt.zero_grad()

    loss = torch.nn.functional.cross_entropy(flat_posteriors, flat_labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_data.parameters(), 10.0)
    opt.step()

    pred_std = idx.to(torch.float32).std()

    if model['stats'] is None:
        model['stats'] = {
            'loss': loss.detach(),
            'count': 1,
            'utts': N,
            'predstd': pred_std,
            'correct': correct_count,
            'valid': valid_count
        }

    else:
        stats = model['stats']
        stats['loss'] += loss.detach()
        stats['count'] += 1
        stats['utts'] += N
        stats['correct'] += correct_count
        stats['predstd'] += pred_std
        stats['valid'] += valid_count

    if (count + 1) % args.save_tick == 0:
        checkpoint.backup()

    if (count + 1) % args.print_tick == 0:
        valid_frames = stats['valid'].cpu()
        batches = stats['count']
        correct_frames = stats['correct'].cpu().to(torch.float32)

        avg_ce = stats['loss'].cpu() / batches
        avg_err = 100.0 - (100.0 * correct_frames / valid_frames)
        tensorboard.add_scalar('%s/ce_loss' % model['name'], avg_ce, step)
        tensorboard.add_scalar('%s/frame_err' % model['name'], avg_err, step)

        if isfirst:
            LOG.info('Summary for step %d [batches seen this run: %d]', step,
                     count + 1)
            tensorboard.add_scalar('utts', stats['utts'] / batches, step)

        LOG.info(
            'Model %s: CE: %f, ERR: %f (CO %0.0f), FRAMES: %d, UTTS: %d, BATCHES: %d PREDSTD: %f',
            model['name'], avg_ce, avg_err, correct_frames, valid_frames,
            stats['utts'], batches, stats['predstd'].cpu() / stats['count'])

        model['stats'] = None
        checkpoint.increment_step()
        # lr_scheduler.step()

if __name__ == '__main__':
    main()
