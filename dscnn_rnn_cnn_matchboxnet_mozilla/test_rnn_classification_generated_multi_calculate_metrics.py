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
import soundfile as sf
# Aux scripts
import kwspipe
import kwscnn
import pdb
import modelconfig

import warnings
warnings.filterwarnings('ignore')


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
                        default=False,
                        help="True for training")
    parser.add_argument("--momentum", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--file_num", type=int, default=None)
    parser.add_argument("--fine_tune", type=bool, default=False)
    parser.add_argument('--version', type=int, default=1)

    args = parser.parse_args()

    if args.words == 'all':
        word_model_name = 'google30'
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
            'levels', 'prior', 'give', 'oil', 'economy', 'cannot', 'parents', 'inside', 'reason', 'houses',
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

        elif args.version == 6:
            args.words = ['land', 'england','process', 'success', 'little', 'battles', 'economy', 'academy','toll',
            'ball', 'ballet', 'light', 'night', 'sight', 'money', 'valley', 'charity', 'figure', 'venture',
            'remain', 'domain', 'empty', 'andy', 'ten', 'tennis', 'inside', 'outside', 'side', 'appear', 'prior']
        else:
            raise ValueError('Incorrect wordset')
    elif args.words == '12class' or args.words == 'sub':
        word_model_name = 'google10'
        args.words = ["yes", "no", "up", "down", "left", "right", "on", "off",
            "stop", "go", "allsilence", "unknown"]
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
        # # modelconfig.RC02(),
        modelconfig.RC03()
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
        MODEL_NAME = 'RNNCNN' + '-LR{}-rank{}-channel{}'.format(0.001, config.rank, config.cnn_channels)

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

        model = kwscnn.DSCNN_RNN_Block(config.cnn_channels,
                        config.rnn_hidden_size,
                        1,
                        device,
                        rank=config.rank, fwd_context=15)

        classifier = kwscnn.Binary_Classification_Block(41,
                                                        100,
                                                        1,
                                                        device,
                                                        islstm=config.islstm,
                                                        isBi=config.isBi,
                                                        dropout=args.dropout,
                                                        num_labels=len(args.words) )

        if args.fine_tune:
            print('***************************')
            print('Phoneme Model : Fine-tuned')
            print('***************************')
        else:
            print('************************')
            print('Phoneme Model : Frozen')
            print('************************')
        print('************************')
        print(f'Version {args.version}')
        print('************************')
        print('*******************************************')
        print(f'Classifier and Pipeline Train Flag: {args.is_training}')
        print('*******************************************')

        model = {
            'name': model.__name__,
            'name_classifier': classifier.__name__,
            'data': model.train(False),
            'classifier': classifier
        }

        model['data'] = model['data'].to(device)
        model['data'] = torch.nn.DataParallel(model['data'])

        model['classifier'] = model['classifier'].to(device)
        model['classifier'] = torch.nn.DataParallel(model['classifier'])


        if args.fine_tune:
            print('*****************')
            print('All Parameters')
            print('*****************')
            total_params = list(model['data'].parameters()) + list(model['classifier'].parameters())
        else:
            print('************************')
            print('Classifier Parameters')
            print('************************')
            total_params = model['classifier'].parameters()

        if args.optim == "adam":
            model['opt'] = torch.optim.Adam(model['data'].parameters(),
                                            lr=config.lr)
            model['opt_classifier'] = torch.optim.Adam(
                total_params, lr=config.lr)
            model['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                model['opt'], T_max=args.train_epochs, eta_min=10e-6)
        #Adding support for SGD optimizer with LR scheduler as Cosine
        if args.optim == "sgd":
            model['opt'] = torch.optim.SGD(model['data'].parameters(),
                                           lr=config.lr)
            model['opt_classifier'] = torch.optim.SGD(
                total_params, lr=config.lr)
            model['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                model['opt'], T_max=args.train_epochs, eta_min=10e-6)

        model['stats'] = None
        model['frame_rate'] = 3  # default frame rate is 3x (30ms)

        model['checkpoint'] = ModelCheckPoint(model['name'],
                                              model_path,
                                              model['data'],
                                              model['opt'],
                                              LOG=LOG)
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
            sum([param.numel() for param in model['data'].parameters()]) +
            sum([param.numel() for param in model['classifier'].parameters()
                 ])) / 1024.0 / 1024.0
        LOG.info('Model %s:  I8 size = %f', model['name'], i16_size)

        tensorboards.append(tensorboard)
        LOGS.append(LOG)

    if args.noise_type == 'noise_batch':
        data_pipe = kwspipe.make_phoneme_google_generated_noise_rir_classification_pipe_test(
            local=args.local, args=args)
    elif args.noise_type == 'additive_rir_noise':
        data_pipe = kwspipe.make_phoneme_google_generated_noise_rir_classification_pipe_test(
            local=args.local, args=args)
    else:
        data_pipe = kwspipe.make_phoneme_google_generated_noise_rir_classification_pipe_test(
            local=args.local, args=args)

    pipe = td.DataPipeLoader(dataset=data_pipe,
                             pin_memory=pin_mem,
                             collate_fn=_noop,
                             num_workers=args.workers)
    # print(type(pipe))
    # print(pipe)

    threads = []
    labels_done = []
    total_number_of_labels = 0
    # dataset = 'google10_test' if len(args.words)==12 else 'google30_test'
    if args.version == 1:
        dataset = 'dataset_1/test'
    elif args.version == 2:
        dataset = 'dataset_2/test'
    elif args.version == 3:
        dataset = 'dataset_3/test'
    elif args.version == 4:
        dataset = 'dataset_4/test'
    elif args.version == 5:
        dataset = 'dataset_5/test'
    else:
        dataset = 'dataset_6/test'
    for k in args.words:
        total_number_of_labels += len(os.listdir(
            f'/mnt/kws_data/mozilla_data/kws_clips_montrealformat/{dataset}/{k}/'))
        # {}/mozilla_data/kws_clips_montrealformat/{}/chunk_data/{}
    print("----------------------------------------------")
    print(args.words)
    print(f"Number of files in this iteration is {total_number_of_labels}")
    print("----------------------------------------------")
    total = np.zeros([len(args.words)])
    wrong_num = 0
    correct_count = np.zeros([len(args.words)])
    
    # Test Script, Gradients can be ignored
    with torch.no_grad():
        for count, item in enumerate(pipe):
            print('Count : ', count)
            # if count > 10 :
            #     break
            feature = item['feature']
            label = np.array(item['label'])[..., 0:-1]
            file_numbers = np.array(item['label'])[..., -1]
            label = torch.tensor(label)
            seqlen = item['seqlen']
            audios = item['audio_samples']
            batch_size = feature.shape[0]
            # pdb.set_trace()

            for i in range(len(models)):
                device = models[i]['device']
                lr_scheduler = models[i]['lr_scheduler']
                
                true_label, pred_label = _process_phoneme_model(models[i], feature.to(device),
                                        label.to(device), seqlen.to(device),
                                        count, args, tensorboards[i], LOGS[i],
                                        True, lr_scheduler, device)
                
                # One hot format
                true_label = true_label.cpu().detach().numpy().reshape(batch_size, -1)
                pred_label = pred_label.cpu().detach().numpy().reshape(batch_size, -1)
                # print(true_label.shape, pred_label.shape)

                # Calculate between indicies
                true_label_idx = np.argmax(true_label, axis=1)
                pred_label_idx = np.argmax(pred_label, axis=1)
                # print(true_label_idx.shape, pred_label_idx.shape)
                
                for j in range(batch_size):
                    if file_numbers[j] in labels_done:
                        continue
                    else:
                        labels_done.append(file_numbers[j])
                        correct_count[true_label_idx[j]] += np.sum(true_label_idx[j] == pred_label_idx[j])
                        total[true_label_idx[j]] += 1
                print(f"Num Labels Completed : {len(labels_done)}")    
            if len(labels_done) == total_number_of_labels:
                break

                

        pred_acc = correct_count*100/total
        print("--------------------------------")
        print(f"Correct : {np.sum(correct_count)}, Total : {np.sum(total)}")
        print(f"Prediction Accuracy : {np.sum(correct_count)*100/np.sum(total)}")
        print("--------------------------------")
        
        for i, w in enumerate(args.words):
            print('Label = {0: <7} ::  Accuracy  = {1} '.format(w, pred_acc[i]))

        for i in range(len(models)):
            tensorboards[i].close()
        print('Test Complete')


def _noop(value):
    return value[0]


def _process_phoneme_model(model, feature, label, seqlen, count, args,
                           tensorboard, LOG, isfirst, lr_scheduler, device):
    # print(f'feature shape {feature.shape}')
    # print(f'label shape {label.shape}')
    # print('--------------------------')
    # import pdb;pdb.set_trace()
    # print("running")
    seqlen_classifier = seqlen.clone() / 3
    model_data = model['data']
    model_data.eval()
    classfier = model['classifier']
    classfier.eval()
    opt = model['opt_classifier']
    checkpoint = model['checkpoint']
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
    posteriors = model_data(feature)
    posteriors = classfier(posteriors, seqlen_classifier)
    N, L, C = posteriors.shape

    flat_posteriors = posteriors.reshape((-1, C))  # to [NL] x C

    assert label.shape == flat_posteriors.shape, \
        "Mis-match in the Shape, Posteriors shape is {} and Label shape is {}".format(flat_posteriors.shape, label.shape)

    # opt.zero_grad()
    # label = label.type(torch.float32)
    # loss = torch.nn.functional.binary_cross_entropy_with_logits(flat_posteriors, label)

    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(classfier.parameters(), 10.0)
    # opt.step()


    flat_posteriors = torch.nn.functional.sigmoid(flat_posteriors)
    # import pdb;pdb.set_trace()
    # flat_pos_sum = torch.sum(flat_posteriors, dim=1)
    # flat_pos_sum = flat_pos_sum.view(-1, * [1]* (flat_posteriors.dim()-1))
    # flat_posteriors = flat_posteriors/flat_pos_sum
    return label, flat_posteriors

if __name__ == '__main__':
    main()
    print('Code Execution Completed\n\n')
    exit()
