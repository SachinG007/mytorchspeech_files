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
import matchboxnet

import modelconfig

import warnings
warnings.filterwarnings('ignore')


def main():
    """Main program entry"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--save_path', default='./rnncnn_models_beta_lessnoise_tts')
    parser.add_argument('--print_tick', type=int, default=500)
    parser.add_argument('--save_tick', type=int, default=10000)
    parser.add_argument('--local', type=bool, default=False)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--mode', default='word')
    parser.add_argument('--words', type=str, required=True)
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
    parser.add_argument("--momentum", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument('--tts', type=bool, default=False)
    parser.add_argument('--synth', type=bool, default=False)
    parser.add_argument('--version', type=int, default='1')

    args = parser.parse_args()
    # print(args.tts)
    # import pdb;pdb.set_trace()
    
    if args.words == 'all':
        if args.version == 1:
            args.words = ['case', 'light', 'songs', 'killed', 'present', 'side', 'word', 'ten', 'buried', 'canada',
          'bank', 'latter', 'floor', 'ships', 'shirt', 'theatre', 'sources', 'mixed', 'simply', 'change',
          'bonds', 'enjoys', 'tail', 'seem', 'compact', 'deaths', 'domain', 'scheme', 'hundred', 'bigger'
            ]
        elif args.version == 2:
            args.words = ['board', 'law', 'states', 'middle', 'night', 'thus', 'england', 'john', 'sons', 'field',
            'cast', 'appear', 'weeks', 'leader', 'dance', 'post', 'engine', 'trains', 'takes', 'making',
            'homer', 'dairy', 'ballet', 'harris', 'parks', 'wild', 'require', 'roberts', 'filming', 'strip'
            ]
        elif args.version == 3:
            args.words = ['days', 'early', 'common', 'william', 'blue', 'success', 'owned', 'modern', 'house', 'king',
            'levels', 'prior', 'give', 'oil', 'economy', 'cannot', 'parents', 'inside', 'reason', 'mixed',
            'grace', 'battles', 'iron', 'rises', 'durham', 'loss', 'eddie', 'georgia', 'varied', 'charity'
            ]
        elif args.version == 4:
            args.words = ['power', 'body', 'parish', 'food', 'popular', 'outside', 'social', 'street', 'little', 'joined',
            'lines', 'leaves', 'thought', 'writing', 'cause', 'ball', 'border', 'stop', 'fish', 'navy',
            'vietnam', 'empty', 'tennis', 'cuts', 'vessel', 'windows', 'replied', 'authors', 'repair', 'matrix'
            ]
        elif args.version == 5:
            args.words = ['land', 'process', 'style', 'role', 'live', 'single', 'york', 'valley', 'french', 'always',
            'table', 'money', 'remain', 'minor', 'albums', 'century', 'academy', 'effects', 'far', 'martin',
            'swiss', 'toll', 'andy', 'figure', 'longest', 'venture', 'revised', 'dean', 'sight', 'join'
            ]

        if args.tts:
            args.words = ['GENERATED_'+w.upper() for w in args.words]
    elif args.words == '12class' or args.words == 'sub':
        
        args.words = ["yes", "no", "up", "down", "left", "right", "on", "off",
            "stop", "go", "allsilence", "unknown"]
        if args.tts:
            args.words = ['GENERATED_'+w.upper() for w in args.words]
    else:
        raise ValueError('Incorrect wordset')

    pin_mem = True

    # Prep the models
    gpu_str = str()
    for gpu in args.gpu.split(','):
        gpu_str = gpu_str + str(gpu) + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    models_config = [
        # modelconfig.RC01(),
        modelconfig.RC03(),
        modelconfig.RC04()
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
        if i==0:
            MODEL_NAME = 'RNNCNN' + '-LR{}-rank{}-channel{}'.format(0.001, config.rank, config.cnn_channels)
        if i==1:
            MODEL_NAME = 'RNNCNN' + '-LR{}-rank{}-channel{}-LSTM'.format(0.001, config.rank, config.cnn_channels)

        # MODEL_NAME = 'RC01' + '-{}'.format(0.001)
        if args.noise_type == 'noise_batch':
            MODEL_NAME += '-NoiseBatch'
        elif args.noise_type == 'additive_rir_noise':
            MODEL_NAME += '-NoiseRirSynthesis'
        # MODEL_NAME = 'Classifier-Generated-' + MODEL_NAME

        MODEL_PATH = args.save_path + '/' + MODEL_NAME

        model_path = MODEL_PATH + '/checkpoints'
        log_path = MODEL_PATH + '/logs'

        model_names.append(MODEL_NAME)
        model_paths.append(model_path)
        log_paths.append(log_path)

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

        LOG.info(args)
        LOG.info(config)
        
        if i==0:
            classifier = matchboxnet.MatchBoxNet(80, num_labels=len(args.words))
        if i==1:
            classifier = matchboxnet.LSTMClassifier(80, 200, 1, device, islstm=True, isBi=True, num_labels=len(args.words))

        model = {
            'name_classifier': classifier.__name__,
            'classifier': classifier
        }
        
        model['classifier'] = model['classifier'].to(device)
        model['classifier'] = torch.nn.DataParallel(model['classifier'])
        
        if args.optim == "adam":
            model['opt_classifier'] = torch.optim.Adam(
                model['classifier'].parameters(), lr=config.lr)
            model['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                model['opt_classifier'], T_max=args.train_epochs, eta_min=10e-6)
        #Adding support for SGD optimizer with LR scheduler as Cosine
        if args.optim == "sgd":
            model['opt_classifier'] = torch.optim.SGD(
               model['classifier'].parameters(), lr=config.lr)
            model['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                model['opt_classifier'], T_max=args.train_epochs, eta_min=10e-6)

        model['stats'] = None
        model['frame_rate'] = 3  # default frame rate is 3x (30ms)

        model['checkpoint_classifier'] = ModelCheckPoint(
            model['name_classifier'],
            model_path,
            model['classifier'],
            model['opt_classifier'],
            LOG=LOG)

        model['device'] = device

        models.append(model)

        os.makedirs(model_path, exist_ok=True)

        i16_size = (
            sum([param.numel() for param in model['classifier'].parameters()
                 ])) / 1024.0 / 1024.0
        LOG.info('Model %s:  I8 size = %f', model['name_classifier'], i16_size)

        tensorboards.append(tensorboard)
        LOGS.append(LOG)

    if args.noise_type == 'noise_batch':
        data_pipe = kwspipe.make_phoneme_google_generated_noise_rir_classification_pipe(
            local=args.local, args=args)
    elif args.noise_type == 'additive_rir_noise':
        data_pipe = kwspipe.make_phoneme_google_generated_noise_rir_classification_pipe(
            local=args.local, args=args)
    else:
        data_pipe = kwspipe.make_phoneme_google_generated_noise_rir_classification_pipe(
            local=args.local, args=args)

    pipe = td.DataPipeLoader(dataset=data_pipe,
                             pin_memory=pin_mem,
                             collate_fn=_noop,
                             num_workers=args.workers)
    # print(type(pipe))
    # print(pipe)

    print(args.words)
    threads = []

    print("************REACHED JUST BEFORE ENUMERATE************")
    for count, item in enumerate(pipe):

        # print(f'item is {item}')
        print(count, flush=True)
        feature = item['feature']
        label = torch.tensor(np.array(item['label']))
        seqlen = item['seqlen']
        audios = item['audio_samples']
                            
        for t in threads:
            t.join()

        threads = []

        for i in range(len(models)):
            device = models[i]['device']
            lr_scheduler = models[i]['lr_scheduler']
            # _process_phoneme_model(models[i], feature.to(device),
            #                        label.to(device), seqlen.to(device), count,
            #                        args, tensorboards[i], LOGS[i], True)

            t = threading.Thread(target=_process_phoneme_model,
                                 args=(models[i], feature.to(device),
                                       label.to(device), seqlen.to(device),
                                       count, args, tensorboards[i], LOGS[i],
                                       True, lr_scheduler, device))
            t.start()
            threads.append(t)

    for i in range(len(models)):
        tensorboards[i].close()
    print('done')


def _noop(value):
    return value[0]


def _process_phoneme_model(model, feature, label, seqlen, count, args,
                           tensorboard, LOG, isfirst, lr_scheduler, device):

    classfier = model['classifier']
    opt = model['opt_classifier']
    checkpoint_classifier = model['checkpoint_classifier']
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

    step = checkpoint_classifier.global_step()
    feature = feature.permute((0, 2, 1))  # NLC to NCL

    posteriors = classfier(feature, seqlen)
    N, C, L = posteriors.shape

    flat_posteriors = posteriors.reshape((-1, C))  # to [NL] x C

    opt.zero_grad()
    label = label.type(torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(flat_posteriors, label)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(classfier.parameters(), 10.0)
    opt.step()

    if model['stats'] is None:
        model['stats'] = {
            'loss': loss.detach(),
            'count': 1,
            'utts': N
        }

    else:
        stats = model['stats']
        stats['loss'] += loss.detach()
        stats['count'] += 1
        stats['utts'] += N

    if (count + 1) % args.save_tick == 0:
        # checkpoint.backup()
        checkpoint_classifier.backup()

    if (count + 1) % args.print_tick == 0:
        batches = stats['count']
        avg_ce = stats['loss'].cpu() / batches
        tensorboard.add_scalar('%s/ce_loss' % model['name_classifier'], avg_ce, step)

        if isfirst:
            LOG.info('Summary for step %d [batches seen this run: %d]', step,
                     count + 1)
            tensorboard.add_scalar('utts', stats['utts'] / batches, step)

        LOG.info(
            'Model %s: CE: %f, UTTS: %d, BATCHES: %d',
            model['name_classifier'], avg_ce,
            stats['utts'], batches)

        model['stats'] = None
        checkpoint_classifier.increment_step()


if __name__ == '__main__':
    main()
