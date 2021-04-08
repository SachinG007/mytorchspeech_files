"""KWS pipeline building module"""
import math
import os
import logging
import json
import numpy as np
import torch
import torchspeech.utils.data as td
from torchspeech.utils.data.kws import KwsBatcher, KwsFeatures, KwsNoiseSynth

LOG = logging.getLogger(__name__)


class _ChunkAudio(td.DataBlock):
    def __init__(self):
        self._item = []
        pass

    def __next__(self):

        if len(self._item) == 0:
            self._item = self.next_input()

        wavslice = self._item[0:96000].copy()
        self._item = self._item[96000:]
        return wavslice


class _KeywordTrimmer(td.DataBlock):
    '''
    This trims the audio after a keyword to ensure it fires in a timely fashion.
    '''
    def __init__(self, segment_len=3.0, pad=0.20, cut_prob=0.3):
        self._pad = pad
        self._segment_len = segment_len
        self._cut_prob = cut_prob

    def __next__(self):

        while True:
            record = self.next_input()
            mlf = record['alignment'].split('\n')

            keyword_start = None
            for line in mlf:
                fields = line.lower().split(' ')
                if len(fields) >= 7 or fields[2].startswith('sil'):
                    word = fields[6] if len(fields) >= 7 else 'sil'
                    if keyword_start is not None:

                        audio = record['audio_samples']
                        desired_samples = math.ceil(self._segment_len * 16000)
                        keyword_end = int(fields[0]) / 10000000.0

                        # For some utts, cut off the audio soon after the keyword end
                        # This is to ensure the model fires in a timely fashion
                        if self._pad > 0 and np.random.random(
                        ) <= self._cut_prob:
                            terminator = keyword_end + self._pad
                            sample_count = int(terminator * 16000)
                            audio = audio[0:sample_count]

                        # Pre-pad with silence to ensure a minimum audio length
                        if len(audio) < desired_samples:
                            prepad_samples = desired_samples - len(audio)
                            pad_pow = np.std(
                                audio[0:prepad_samples]) * 0.01 + 1e-6
                            padding = np.random.normal(size=(
                                prepad_samples)).astype(audio.dtype) * pad_pow
                            audio = np.concatenate((padding, audio))
                        else:
                            prepad_samples = 0

                        frame_end = int(keyword_end * 100) + int(
                            prepad_samples / 160.0)
                        label = np.zeros((len(record['audio_samples'])),
                                         dtype=np.int64)
                        label[frame_end - 6:frame_end + 18] = -100
                        label[frame_end:frame_end + 12] = 1

                        record['label'] = label
                        record['audio_samples'] = audio
                        record['ig'] = True
                        return record

                    if word == 'cortana':
                        keyword_start = int(fields[0]) / 10000000.0

            # else try next record...


class _KeywordRemover(td.DataBlock):
    '''
    This trims the audio after a keyword to ensure it fires in a timely fashion.
    '''
    def __init__(self):
        pass

    def __next__(self):

        while True:
            record = self.next_input()

            # Doesn't contain cortana, so pass it through
            if 'cortana' in record['transcription'].lower():
                return record

            mlf = record['alignment'].split('\n')

            keyword_start = None
            for line in mlf:
                fields = line.lower().split(' ')
                if len(fields) >= 7 or fields[2].startswith('sil'):
                    word = fields[6] if len(fields) >= 7 else 'sil'
                    if keyword_start is not None:

                        audio = record['audio_samples']
                        desired_samples = math.ceil(self._segment_len * 16000)
                        keyword_end = int(fields[0]) / 10000000.0

                        # For some utts, cut off the audio soon after the keyword end
                        # This is to ensure the model fires in a timely fashion
                        if self._pad > 0 and np.random.random(
                        ) <= self._cut_prob:
                            terminator = keyword_end + self._pad
                            sample_count = int(terminator * 16000)
                            audio = audio[0:sample_count]

                        # Pre-pad with silence to ensure a minimum audio length
                        if len(audio) < desired_samples:
                            prepad_samples = desired_samples - len(audio)
                            pad_pow = np.std(
                                audio[0:prepad_samples]) * 0.01 + 1e-6
                            padding = np.random.normal(size=(
                                prepad_samples)).astype(audio.dtype) * pad_pow
                            audio = np.concatenate((padding, audio))
                        else:
                            prepad_samples = 0

                        frame_end = int(keyword_end * 100) + int(
                            prepad_samples / 160.0)
                        label = np.zeros((len(record['audio_samples'])),
                                         dtype=np.int64)
                        label[frame_end - 6:frame_end + 18] = -100
                        label[frame_end:frame_end + 12] = 1

                        record['label'] = label
                        record['audio_samples'] = audio
                        record['ig'] = True
                        return record

                    if word == 'cortana':
                        keyword_start = int(fields[0]) / 10000000.0

            # else try next record...


class _MergeStreams(td.ZipStreams):
    def __init__(self, debug, ig, oog):
        # func = _MergeStreams._post_op_debug if debug else _MergeStreams._post_op
        func = _MergeStreams._post_op_debug
        super().__init__((ig, oog), func)

    @staticmethod
    def _post_op(ig, oog):
        return {'ig_feature': ig['feature'], 'oog_feature': oog['feature']}

    @staticmethod
    def _post_op_debug(ig, oog):
        item = {}
        for key, value in ig.items():
            item['ig_%s' % key] = value
        for key, value in oog.items():
            item['oog_%s' % key] = value

        return item


def _check_ig_ok(field):
    return "cortana" in field['transcription'].lower() and len(
        field['alignment']) > 10


def _check_oog_ok(field):
    return "cortana" not in field['transcription'].lower()


def _check_igculler(field):
    return "cortana" not in field['transcription'].lower() or len(
        field['alignment']) > 10


def _check_16k(field):
    return field['sr'] == 16000


def _spoof_oog_labels(field):
    frame_count = field['feature'].shape[0]
    labels = np.zeros((frame_count), dtype=np.int64)
    field['label'] = labels
    field['ig'] = False
    return field


def _extract_frame_count(item):
    return -item['feature'].shape[0]

def _extract_label(item):
    return item['label'][0]

def _check_utt_length(item):
    return item['feature'].shape[0] >= 250


def _map_to_audio(item):
    return {'audio': item}


def _set_min_audiolen(item):

    min_samples = 3 * 16000
    audio = item['audio_samples']
    if len(audio) >= min_samples:
        return item

    prepad_samples = min_samples - len(audio)
    pad_pow = np.std(audio[0:prepad_samples]) * 0.01 + 1e-6
    padding = np.random.normal(size=(prepad_samples)).astype(
        audio.dtype) * pad_pow
    item['audio_samples'] = np.concatenate((padding, audio))

    return item


def _create_oog_pipelet(local, noises, args):

    if local:
        table_path = '../80dim.table'
        chunk_meta = list(
            td.BinaryChunkLister('/home/anthony/data/se_chunks/miniset.json',
                                 '/home/anthony/data/se_chunks/'))
    else:
        table_path = '{}/data/80dim.table'.format(args.datablob)
        chunk_meta = list(
            td.BinaryChunkLister(
                '{}/data/chunk_data/file_set.json'.format(args.datablob),
                '{}/data/chunk_data/'.format(args.datablob)))

    speech_oog = td.DataPipe(
        td.IIDListSampler(chunk_meta, sharding=False),
        td.BinaryChunkDeserializer(feat_dim=80), td.Filter(_check_oog_ok),
        td.DecodeWave('audio', 'audio_samples'), td.Filter(_check_16k, 100000),
        td.KeepFields('audio_samples', 'alignment', 'transcription', 'info'),
        td.Transform(_set_min_audiolen),
        KwsFeatures('audio_samples',
                    table_path=table_path,
                    output_field='feature'), td.Filter(_check_utt_length),
        td.Transform(_spoof_oog_labels))

    noise_oog = td.DataPipe(
        noises, _ChunkAudio(), td.Transform(_map_to_audio),
        KwsFeatures('audio', table_path=table_path, output_field='feature'),
        td.Filter(_check_utt_length), td.Transform(_spoof_oog_labels))

    return td.Combiner([(speech_oog, 3), (noise_oog, 1)])


def _scan_hdf5(root):
    chunk_paths = []
    for subdir in os.listdir(root):
        manifest = os.path.join(root, subdir, 'chunks.json')
        if os.path.exists(manifest):
            with open(manifest, 'r') as f:
                info = json.load(f)
                for chunk in info['chunks']:
                    chunk_paths.append(os.path.join(root, subdir, chunk))

    return chunk_paths


def _get_cortana_pipelets(local, noises, rirs, args):

    if local:
        cortana_chunks = ['../B000.hdf5', '../B000.hdf5']
        hc_chunks = ['../B069.hdf5', '../B069.hdf5']
        table_path = '../80dim.table'
    else:
        table_path = '{}/data/80dim.table'.format(args.datablob)
        cortana_root = '{}/data/data/logs/cortana/'.format(args.datablob)
        hc_root = '{}/data/data/logs/hey_cortana/'.format(args.datablob)

        cortana_chunks = _scan_hdf5(cortana_root)
        if not cortana_chunks:
            raise ValueError('Did not find any valid cortana chunks')

        LOG.info('Scanned for cortana chunks, found %d', len(cortana_chunks))

        hc_chunks = _scan_hdf5(hc_root)
        if not cortana_chunks:
            raise ValueError('Did not find any valid hey cortana chunks')

        LOG.info('Scanned for hey cortana chunks, found %d', len(hc_chunks))

    synth_block = KwsNoiseSynth('audio_samples',
                                'synthaudio',
                                noises,
                                rirs,
                                snr_samples=[-5, 0, 0, 0, 5, 10, 15, 100, 100],
                                rir_chance=0.25,
                                synth_chance=0.5,
                                noise_pooling=50,
                                wgn_snr_samples=[5, 10, 15, 100, 100])

    pipe1 = _create_ig_pipelet(cortana_chunks, table_path)
    pipe2 = _create_ig_pipelet(hc_chunks, table_path)

    # Returns a 50/50 mix of 'cortana' and 'hey cortana'
    return td.DataPipe(
        td.Combiner([(pipe1, 1), (pipe2, 1)]), synth_block,
        KwsFeatures('synthaudio',
                    table_path=table_path,
                    output_field='feature'), td.Filter(_check_utt_length))


def _create_ig_pipelet(data_chunks, table_path):

    return td.DataPipe(td.IIDListSampler(data_chunks, sharding=False),
                       td.HDF5Deserializer(), td.Filter(_check_ig_ok, 100),
                       td.DecodeWave('binarycontent', 'audio_samples'),
                       td.Filter(_check_16k, 100000),
                       _KeywordTrimmer(3.0, 0.20, 1.0))


def _create_igcull_pipelet(data_chunks, table_path):

    return td.DataPipe(td.IIDListSampler(data_chunks, sharding=False),
                       td.HDF5Deserializer(), td.Filter(_check_igculler, 100),
                       td.DecodeWave('binarycontent', 'audio_samples'),
                       td.Filter(_check_16k, 100000), _KeywordRemover())


def _ig_lab_to_numpy(field):
    field['ig'] = np.array(field['ig'], dtype=np.bool)
    return field


def get_hdf5_noise(hdf5_record):
    return hdf5_record['samples']


def make_mixed_pipe(local, args):

    if local:
        sort_batch_size = 1000
        random_buffer = 15
        batch_frames = 7500
        list_fields = ['ig']

        additive_noises = []
        for root, _, filenames in os.walk(
                '/home/anthony/data/additive_noise/'):
            additive_noises.extend(
                (os.path.join(root, name) for name in filenames))

        rir_noises = []
        for root, _, filenames in os.walk('/home/anthony/data/iir/'):
            rir_noises.extend((os.path.join(root, name) for name in filenames))

    else:

        additive_noises = []
        for root, _, filenames in os.walk(
                '{}/data/data/noises/additive/'.format(args.datablob)):
            additive_noises.extend(
                (os.path.join(root, name) for name in filenames))

        rir_noises = []
        for root, _, filenames in os.walk('{}/data/data/noises/iir/'.format(
                args.datablob)):
            rir_noises.extend((os.path.join(root, name) for name in filenames))

        sort_batch_size = 7500
        random_buffer = 100
        batch_frames = 60000
        list_fields = ['ig']

    LOG.info('Found %d additive noise chunks', len(additive_noises))
    LOG.info('Found %d iir chunks', len(rir_noises))

    noise_pipe = td.DataPipe(td.IIDListSampler(additive_noises),
                             td.HDF5Deserializer(),
                             td.Transform(get_hdf5_noise))

    rir_pipe = td.DataPipe(td.IIDListSampler(rir_noises),
                           td.HDF5Deserializer(), td.Transform(get_hdf5_noise),
                           td.RandomizeBuffer(100))

    ig_pipe = _get_cortana_pipelets(local, noise_pipe, rir_pipe, args)
    oog_pipe = _create_oog_pipelet(local, noise_pipe, args)

    return td.DataPipe(
        td.Combiner([(ig_pipe, 1), (oog_pipe, 5)], mode='shuffle'),
        td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
        td.PrefetchBuffer(),
        KwsBatcher('feature',
                   'label',
                   ignore_label=0,
                   list_fields=list_fields,
                   max_frame_count=batch_frames,
                   pad_front=True), td.RandomizeBuffer(random_buffer),
        td.Transform(_ig_lab_to_numpy),
        td.ConvertToTensors('feature', 'label', 'ig', 'seqlen'))


def make_phoneme_pipe(local, args):

    if local:
        phones_file = '/home/anthony/work/cortanaWW/phones.txt'
        sort_batch_size = 1000
        random_buffer = 15
        batch_frames = 5000
        table_path = '../80dim.table'
        chunk_meta = list(
            td.BinaryChunkLister('/home/anthony/data/se_chunks/miniset.json',
                                 '/home/anthony/data/se_chunks/'))
    else:
        phones_file = '{}/data/phones.txt'.format(args.datablob)
        sort_batch_size = 5000
        random_buffer = 100
        batch_frames = 1600 * args.batch_size
        table_path = '{}/data/80dim.table'.format(args.datablob)
        chunk_meta = list(
            td.BinaryChunkLister(
                '{}/data/chunk_data/{}'.format(args.datablob, args.file_set),
                '{}/data/chunk_data/'.format(args.datablob)))

    # return td.DataPipe(
    #     td.IIDListSampler(chunk_meta, sharding=False),
    #     td.BinaryChunkDeserializer(feat_dim=80),
    #     td.KeepFields('feature', 'label'), td.Filter(_check_utt_length),
    #     td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
    #     td.PrefetchBuffer(),
    #     KwsBatcher('feature',
    #                'label',
    #                ignore_label=-100,
    #                max_frame_count=batch_frames,
    #                list_fields=[]), td.RandomizeBuffer(random_buffer),
    #     td.ConvertToTensors('feature', 'label', 'seqlen'))

    # return td.DataPipe(
    #     td.IIDListSampler(chunk_meta, sharding=False),
    #     td.BinaryChunkDeserializer(feat_dim=80),
    #     td.DecodeWave('audio', 'audio_samples'), td.Filter(_check_16k, 100000),
    #     td.KeepFields('audio_samples', 'alignment', 'transcription', 'info',
    #                   'feature', 'label'),
    #     KwsFeatures('audio_samples',
    #                 table_path=table_path,
    #                 output_field='feature2'), td.Filter(_check_utt_length),
    #     td.kws.KwsExtractMonophones(phones_file,
    #                                 'alignment',
    #                                 output_field='label2'),
    #     td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
    #     td.PrefetchBuffer(),
    #     KwsBatcher('feature',
    #                'label',
    #                ignore_label=-100,
    #                max_frame_count=batch_frames,
    #                list_fields=[
    #                    'audio_samples', 'info', 'alignment', 'transcription',
    #                    'feature2', 'label2'
    #                ]), td.RandomizeBuffer(random_buffer),
    #     td.ConvertToTensors('feature', 'label', 'seqlen'))

    return td.DataPipe(
        td.IIDListSampler(chunk_meta,
                          sharding=False,
                          gpu_rank=args.gpu,
                          is_training=args.is_training),
        td.BinaryChunkDeserializer(feat_dim=80),
        td.DecodeWave('audio', 'audio_samples'), td.Filter(_check_16k, 100000),
        td.KeepFields('audio_samples', 'alignment', 'transcription', 'info'),
        KwsFeatures('audio_samples',
                    table_path=table_path,
                    output_field='feature'), td.Filter(_check_utt_length),
        td.kws.KwsExtractMonophones(phones_file,
                                    'alignment',
                                    output_field='label'),
        td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
        td.PrefetchBuffer(),
        KwsBatcher('feature',
                   'label',
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   list_fields=[
                       'audio_samples', 'info', 'alignment', 'transcription'
                   ]), td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'label', 'seqlen'))


def make_noise_phoneme_pipe(local, args):

    if local:
        phones_file = '/home/anthony/work/cortanaWW/phones.txt'
        sort_batch_size = 1000
        random_buffer = 15
        batch_frames = 5000
        table_path = '../80dim.table'
        chunk_meta = list(
            td.BinaryChunkLister('/home/anthony/data/se_chunks/miniset.json',
                                 '/home/anthony/data/se_chunks/'))
    else:
        phones_file = '{}/data/phones.txt'.format(args.datablob)
        sort_batch_size = 5000
        random_buffer = 100
        batch_frames = 1600 * args.batch_size
        table_path = '{}/data/80dim.table'.format(args.datablob)
        chunk_meta = list(
            td.BinaryChunkLister(
                '{}/data/chunk_data/{}'.format(args.datablob, args.file_set),
                '{}/data/chunk_data/'.format(args.datablob)))

        additive_noises = []
        for root, _, filenames in os.walk('{}/data/noises/additive/'.format(
                args.datablob)):
            additive_noises.extend(
                (os.path.join(root, name) for name in filenames))

    LOG.info('Found %d additive noise chunks', len(additive_noises))
    print("Length of additive noises", len(additive_noises))

    noise_pipe = td.DataPipe(td.IIDListSampler(additive_noises),
                             td.HDF5Deserializer(),
                             td.Transform(get_hdf5_noise))

    speech_oog = td.DataPipe(
        td.IIDListSampler(chunk_meta,
                          sharding=False,
                          gpu_rank=args.gpu,
                          is_training=args.is_training),
        td.BinaryChunkDeserializer(feat_dim=80), td.Filter(_check_oog_ok),
        td.DecodeWave('audio', 'audio_samples'), td.Filter(_check_16k, 100000),
        td.KeepFields('audio_samples', 'alignment', 'transcription', 'info'),
        td.Transform(_set_min_audiolen),
        KwsFeatures('audio_samples',
                    table_path=table_path,
                    output_field='feature'), td.Filter(_check_utt_length),
        td.kws.KwsExtractMonophones(phones_file,
                                    'alignment',
                                    output_field='label'))

    noise_oog = td.DataPipe(
        noise_pipe, _ChunkAudio(), td.Transform(_map_to_audio),
        KwsFeatures('audio', table_path=table_path, output_field='feature'),
        td.Filter(_check_utt_length), td.Transform(_spoof_oog_labels))

    oog_pipe = td.Combiner([(speech_oog, 3), (noise_oog, 1)])

    return td.DataPipe(
        oog_pipe, td.BasicBatch(sort_batch_size,
                                sort_func=_extract_frame_count),
        td.PrefetchBuffer(),
        KwsBatcher('feature',
                   'label',
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   list_fields=[]), td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'label', 'seqlen'))


def make_noise_rir_synthesis_phoneme_pipe(local, args):

    if local:
        phones_file = '/home/anthony/work/cortanaWW/phones.txt'
        sort_batch_size = 1000
        random_buffer = 15
        batch_frames = 5000
        table_path = '../80dim.table'
        chunk_meta = list(
            td.BinaryChunkLister('/home/anthony/data/se_chunks/miniset.json',
                                 '/home/anthony/data/se_chunks/'))
    else:
        phones_file = '{}/data/phones.txt'.format(args.datablob)
        sort_batch_size = 5000
        random_buffer = 100
        batch_frames = 1600 * args.batch_size
        table_path = '{}/data/80dim.table'.format(args.datablob)
        chunk_meta = list(
            td.BinaryChunkLister(
                '{}/data/chunk_data/{}'.format(args.datablob, args.file_set),
                '{}/data/chunk_data/'.format(args.datablob)))

        additive_noises = []
        for root, _, filenames in os.walk('{}/data/noises/additive/'.format(
                args.datablob)):
            additive_noises.extend(
                (os.path.join(root, name) for name in filenames))

        rir_noises = []
        for root, _, filenames in os.walk('{}/data/noises/iir/'.format(
                args.datablob)):
            rir_noises.extend((os.path.join(root, name) for name in filenames))

    LOG.info('Found %d additive noise chunks', len(additive_noises))
    LOG.info('Found %d iir chunks', len(rir_noises))

    noise_pipe = td.DataPipe(td.IIDListSampler(additive_noises),
                             td.HDF5Deserializer(),
                             td.Transform(get_hdf5_noise))

    rir_pipe = td.DataPipe(td.IIDListSampler(rir_noises),
                           td.HDF5Deserializer(), td.Transform(get_hdf5_noise),
                           td.RandomizeBuffer(100))

    synth_block = KwsNoiseSynth('audio_samples',
                                'synthaudio',
                                noise_pipe,
                                rir_pipe,
                                snr_samples=[-5, 0, 0, 0, 5, 10, 15, 100, 100],
                                rir_chance=0.25,
                                synth_chance=0.5,
                                noise_pooling=50,
                                wgn_snr_samples=[5, 10, 15, 100, 100])

    stci_pos = list(
        td.BinaryChunkLister(
            '/mnt/kws_data/data/MSRI/STCI/chunk_data/POSITIVE/file_set_train.json',
            '/mnt/kws_data/data/MSRI/STCI/chunk_data/POSITIVE/'))
    
    data_pipes = []
    
    data_pipe3 = td.DataPipe(
                td.IIDListSampler(stci_pos,
                                sharding=False,
                                gpu_rank=args.gpu,
                                is_training=args.is_training),
                td.BinaryChunkDeserializer(feat_dim=80))

    data_pipes.append(data_pipe3)

    combined_pipe = td.Combiner([(pp, 1) for pp in data_pipes])
    # return td.DataPipe(
    #     td.IIDListSampler(chunk_meta,
    #                       sharding=False,
    #                       gpu_rank=args.gpu,
    #                       is_training=args.is_training),
    #     td.BinaryChunkDeserializer(feat_dim=80),
    #     td.DecodeWave('audio', 'audio_samples'), td.Filter(_check_16k, 100000),
    #     td.KeepFields('audio_samples', 'alignment', 'transcription', 'info'),
    #     synth_block,
    #     KwsFeatures('synthaudio',
    #                 table_path=table_path,
    #                 output_field='feature'), td.Filter(_check_utt_length),
    #     td.kws.KwsExtractMonophones(phones_file,
    #                                 'alignment',
    #                                 output_field='label'),
    #     td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
    #     td.PrefetchBuffer(),
    #     KwsBatcher('feature',
    #                'label',
    #                ignore_label=-100,
    #                max_frame_count=batch_frames,
    #                list_fields=[
    #                    'audio_samples', 'info', 'alignment', 'transcription'
    #                ]), td.RandomizeBuffer(random_buffer),
    #     td.ConvertToTensors('feature', 'label', 'seqlen'))
    # pad_value = KwsFeatures('audio_samples',
    #                         table_path=table_path,
    #                         output_field='feature').get_silence_features()



    return td.DataPipe(
        combined_pipe, td.DecodeWave('audio', 'audio_samples'),
        td.Filter(_check_16k, 100000), td.KeepFields('audio_samples', 'label'),
        KwsFeatures('audio_samples',
                    table_path=table_path,
                    output_field='feature'),
        td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
        td.PrefetchBuffer(),
        KwsBatcher('feature',
                   None,
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   list_fields=['label', 'audio_samples']), td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'seqlen'))

def make_phoneme_cortana_pipe(local, args):

    phones_file = '{}/data/phones.txt'.format(args.datablob)
    sort_batch_size = 5000
    random_buffer = 100
    batch_frames = 1600 * args.batch_size
    table_path = '{}/data/80dim.table'.format(args.datablob)
    chunk_meta = list(
        td.BinaryChunkLister(
            '{}/data/MSRI/STCI/chunk_data/{}'.format(args.datablob,
                                                     args.file_set),
            '{}/data/MSRI/STCI/chunk_data/'.format(args.datablob)))

    return td.DataPipe(
        td.IIDListSampler(chunk_meta,
                          sharding=False,
                          gpu_rank=args.gpu,
                          is_training=args.is_training),
        td.BinaryChunkDeserializer(feat_dim=80),
        td.DecodeWave('audio', 'audio_samples'), td.Filter(_check_16k, 100000),
        td.KeepFields('audio_samples'),
        KwsFeatures('audio_samples',
                    table_path=table_path,
                    output_field='feature'),
        td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
        td.PrefetchBuffer(),
        KwsBatcher('feature',
                   None,
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   list_fields=['audio_samples']),
        td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'seqlen'))


def make_phoneme_cortana_generated_noise_rir_classification_pipe_test(
    local, args):

    phones_file = '{}/data/phones.txt'.format(args.datablob)
    sort_batch_size = 5000
    random_buffer = 100
    batch_frames = 1600 * args.batch_size
    table_path = '{}/data/80dim.table'.format(args.datablob)

    chunk_metas = []

    # if os.path.isfile('{}/data/MSRI/STCI/chunk_data/POSITIVE/{}'.format(
    #         args.datablob, args.file_set)):

    #     positive_chunk_meta = list(
    #         td.BinaryChunkLister(
    #             '{}/data/MSRI/STCI/chunk_data/POSITIVE/{}'.format(
    #                 args.datablob, args.file_set),
    #             '{}/data/MSRI/STCI/chunk_data/POSITIVE'.format(args.datablob)))

    #     chunk_metas.append(positive_chunk_meta)

    # if os.path.isfile('{}/data/MSRI/STCI/chunk_data/NEGATIVE/{}'.format(
    #         args.datablob, args.file_set)):
    #     negative_chunk_meta = list(
    #         td.BinaryChunkLister(
    #             '{}/data/MSRI/STCI/chunk_data/NEGATIVE/{}'.format(
    #                 args.datablob, args.file_set),
    #             '{}/data/MSRI/STCI/chunk_data/NEGATIVE'.format(args.datablob)))

    #     chunk_metas.append(negative_chunk_meta)

    if os.path.isfile('{}/data/MSRI/SGY_CMDS/chunk_data/ajay/hello_detect{}/{}'.format(
            args.datablob, args.file_num, args.file_set)):

        hey_chunk_meta = list(
            td.BinaryChunkLister(
                '{}/data/MSRI/SGY_CMDS/chunk_data/ajay/hello_detect{}/{}'.format(
                    args.datablob, args.file_num, args.file_set),
                '{}/data/MSRI/SGY_CMDS/chunk_data/ajay/hello_detect{}'.format(args.datablob, args.file_num)))

        chunk_metas.append(hey_chunk_meta)

    # if os.path.isfile('{}/data/MSRI/STCI/chunk_data/NEGATIVE/{}'.format(
    #         args.datablob, args.file_set)):
    #     negative_chunk_meta = list(
    #         td.BinaryChunkLister(
    #             '{}/data/MSRI/STCI/chunk_data/NEGATIVE/{}'.format(
    #                 args.datablob, args.file_set),
    #             '{}/data/MSRI/STCI/chunk_data/NEGATIVE'.format(args.datablob)))

    #     chunk_metas.append(negative_chunk_meta)
        
    data_pipes = [
        td.DataPipe(
            td.ListSampler(kkk,
                           sharding=False,
                           gpu_rank=args.gpu,
                           is_training=True),
            td.BinaryChunkDeserializer(feat_dim=80)) for kkk in chunk_metas
    ]

    combined_pipe = td.Combiner([(pp, 1) for pp in data_pipes], mode='test')

    pad_value = KwsFeatures('audio_samples',
                            table_path=table_path,
                            output_field='feature').get_silence_features()

    return td.DataPipe(
        combined_pipe, td.DecodeWave('audio', 'audio_samples'),
        td.Filter(_check_16k, 100000), td.KeepFields('audio_samples', 'label'),
        KwsFeatures('audio_samples',
                    table_path=table_path,
                    output_field='feature'),
        td.BasicBatch(sort_batch_size, sort_func=_extract_label),
        td.PrefetchBuffer(),
        KwsBatcher('feature',
                   None,
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   use_single_batch = False,
                   list_fields=['label', 'audio_samples']), td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'seqlen'))


def make_phoneme_google_generated_noise_rir_classification_pipe(local, args):

    phones_file = '{}/data/phones.txt'.format(args.datablob)
    sort_batch_size = 5000
    random_buffer = 100
    batch_frames = 1600 * args.batch_size
    table_path = '{}/data/80dim.table'.format(args.datablob)

    additive_noises = []
    for root, _, filenames in os.walk('{}/data/noises/additive/'.format(
            args.datablob)):
        additive_noises.extend(
            (os.path.join(root, name) for name in filenames))

    rir_noises = []
    for root, _, filenames in os.walk('{}/data/noises/iir/'.format(
            args.datablob)):
        rir_noises.extend((os.path.join(root, name) for name in filenames))

    print(f'Found {len(additive_noises)} additive noise chunks')
    print(f'Found {len(rir_noises)} iir chunks')

    noise_pipe = td.DataPipe(td.IIDListSampler(additive_noises),
                             td.HDF5Deserializer(),
                             td.Transform(get_hdf5_noise))

    rir_pipe = td.DataPipe(td.IIDListSampler(rir_noises),
                           td.HDF5Deserializer(), td.Transform(get_hdf5_noise),
                           td.RandomizeBuffer(100))

    if args.synth:
        print('*******\nSynth\n*******')
        synth_block = KwsNoiseSynth('audio_samples',
                                'synthaudio',
                                noise_pipe,
                                rir_pipe,
                                snr_samples=[-5, 0, 0, 5, 10, 15, 40, 100, 100],
                                rir_chance=0.9,
                                synth_chance=0.9,
                                noise_pooling=50,
                                wgn_snr_samples=[5, 10, 20, 40, 60])
    else:
        print('**********\nNo Synth\n**********')
        synth_block = KwsNoiseSynth('audio_samples',
                                    'synthaudio',
                                    noise_pipe,
                                    rir_pipe,
                                    snr_samples=[-5, 0, 0, 5, 10, 15, 40, 100, 100],
                                    rir_chance=0,
                                    synth_chance=0,
                                    noise_pooling=50,
                                    wgn_snr_samples=[5, 10, 20, 40, 60])

    chunk_metas = []

    if args.tts:
        print('**********')
        print('TTS Data')
        print('**********')

        words = args.words
        for word in words:
            word = word.upper()
            generated_positive_chunk_meta = list(
                td.BinaryChunkLister(
                    '{}/mozilla_data/kws_clips_montrealformat/dataset_1_azure_tts/chunk_data/{}/{}'
                    .format(args.datablob, word, args.file_set),
                    '{}/mozilla_data/kws_clips_montrealformat/dataset_1_azure_tts/chunk_data/{}'
                    .format(args.datablob, word)))
            chunk_metas.append(generated_positive_chunk_meta)

        for word in words:
            word = word.upper()
            generated_positive_chunk_meta = list(
                td.BinaryChunkLister(
                    '{}/mozilla_data/kws_clips_montrealformat/dataset_1_google_tts/chunk_data/{}/{}'
                    .format(args.datablob, word, args.file_set),
                    '{}/mozilla_data/kws_clips_montrealformat/dataset_1_google_tts/chunk_data/{}'
                    .format(args.datablob, word)))
            chunk_metas.append(generated_positive_chunk_meta)
    
    else:
        print('************')
        print('Normal Data')
        print('************')
        
        words = args.words
        for word in words:
            generated_positive_chunk_meta = list(
                td.BinaryChunkLister(
                    '{}/mozilla_data/kws_clips_montrealformat/dataset_1/train/chunk_data/{}/{}'
                    .format(args.datablob, word, args.file_set),
                    '{}/mozilla_data/kws_clips_montrealformat/dataset_1/train/chunk_data/{}'
                    .format(args.datablob, word)))
            chunk_metas.append(generated_positive_chunk_meta)
    
    print("Total Number of Word chunk folders : ", len(chunk_metas))

    data_pipes = [
        td.DataPipe(
            td.IIDListSampler(kkk,
                              sharding=False,
                              gpu_rank=args.gpu,
                              is_training=args.is_training),
            td.BinaryChunkDeserializer(feat_dim=80)) for kkk in chunk_metas
    ]

    combined_pipe = td.Combiner([(pp, 1) for pp in data_pipes])

    pad_value = KwsFeatures('audio_samples',
                            table_path=table_path,
                            output_field='feature').get_silence_features()

    return td.DataPipe(
        combined_pipe, td.DecodeWave('audio', 'audio_samples'),
        td.Filter(_check_16k, 100000), td.KeepFields('audio_samples', 'label'),
        synth_block,
        KwsFeatures('synthaudio',
                    table_path=table_path,
                    output_field='feature'),
        td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
        KwsBatcher('feature',
                   None,
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   list_fields=['label', 'audio_samples'],
                   ), td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'seqlen'))

def make_phoneme_google_generated_noise_rir_classification_pipe_test(local, args):

    phones_file = '{}/data/phones.txt'.format(args.datablob)
    sort_batch_size = 5000
    random_buffer = 100
    batch_frames = 1600 * args.batch_size
    table_path = '{}/data/80dim.table'.format(args.datablob)

    additive_noises = []
    for root, _, filenames in os.walk('{}/data/noises/additive/'.format(
            args.datablob)):
        additive_noises.extend(
            (os.path.join(root, name) for name in filenames))

    rir_noises = []
    for root, _, filenames in os.walk('{}/data/noises/iir/'.format(
            args.datablob)):
        rir_noises.extend((os.path.join(root, name) for name in filenames))

    print(f'Found {len(additive_noises)} additive noise chunks')
    print(f'Found {len(rir_noises)} iir chunks')

    noise_pipe = td.DataPipe(td.IIDListSampler(additive_noises),
                             td.HDF5Deserializer(),
                             td.Transform(get_hdf5_noise))

    rir_pipe = td.DataPipe(td.IIDListSampler(rir_noises),
                           td.HDF5Deserializer(), td.Transform(get_hdf5_noise),
                           td.RandomizeBuffer(100))

    chunk_metas = []

    dataset = 'google10_test' if len(args.words)==12 else 'google30_test'
    if args.version == '2' or args.version == 'v2':
        dataset += '_v2'
    print(dataset)
    words = args.words
    for word in words:
        word = word.lower()
        generated_positive_chunk_meta = list(
            td.BinaryChunkLister(
                '{}/data/{}/chunk_data/{}/{}'
                .format(args.datablob, dataset, word, args.file_set),
                '{}/data/{}/chunk_data/{}'
                .format(args.datablob, dataset, word)))
        chunk_metas.append(generated_positive_chunk_meta)

    # if os.path.isfile('{}/data/MSRI/SGY_CMDS/chunk_data/ajay/cmd_detect{}/{}'.format(
    #         args.datablob, args.file_num, args.file_set)):

    #     hey_chunk_meta = list(
    #         td.BinaryChunkLister(
    #             '{}/data/MSRI/SGY_CMDS/chunk_data/ajay/cmd_detect{}/{}'.format(
    #                 args.datablob, args.file_num, args.file_set),
    #             '{}/data/MSRI/SGY_CMDS/chunk_data/ajay/cmd_detect{}'.format(args.datablob, args.file_num)))

    #     chunk_metas.append(hey_chunk_meta)

    print("Total Different Words Folder: ", len(chunk_metas))
    data_pipes = [
        td.DataPipe(
            td.IIDListSampler(kkk,
                              sharding=False,
                              gpu_rank=args.gpu,
                              is_training=args.is_training),
            td.BinaryChunkDeserializer(feat_dim=80)) for kkk in chunk_metas
    ]
    
    combined_pipe = td.Combiner([(pp, 1) for pp in data_pipes], mode='stochastic' if args.is_training else 'test')
    
    pad_value = KwsFeatures('audio_samples',
                            table_path=table_path,
                            output_field='feature').get_silence_features()

    return td.DataPipe(
        combined_pipe, td.DecodeWave('audio', 'audio_samples'),
        td.Filter(_check_16k, 100000), td.KeepFields('audio_samples', 'label'),
        KwsFeatures('audio_samples',
                    table_path=table_path,
                    output_field='feature'),
        # td.BasicBatch(sort_batch_size, sort_func=_extract_label),
        td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
        KwsBatcher('feature',
                   None,
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   use_single_batch = False,
                   list_fields=['label', 'audio_samples'],
                   pad_value=pad_value), td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'seqlen'))



def make_noise_rir_synthesis_phoneme_pipe_train(local, args):

    if local:
        phones_file = '/home/anthony/work/cortanaWW/phones.txt'
        sort_batch_size = 1000
        random_buffer = 15
        batch_frames = 5000
        table_path = '../80dim.table'
        chunk_meta = list(
            td.BinaryChunkLister('/home/anthony/data/se_chunks/miniset.json',
                                 '/home/anthony/data/se_chunks/'))
    else:
        phones_file = '{}/data/phones.txt'.format(args.datablob)
        sort_batch_size = 5000
        random_buffer = 100
        batch_frames = 1600 * args.batch_size
        table_path = '{}/data/80dim.table'.format(args.datablob)
        chunk_meta = list(
            td.BinaryChunkLister(
                '{}/data/chunk_data/{}'.format(args.datablob, args.file_set),
                '{}/data/chunk_data/'.format(args.datablob)))

        additive_noises = []
        for root, _, filenames in os.walk('{}/data/noises/additive/'.format(
                args.datablob)):
            additive_noises.extend(
                (os.path.join(root, name) for name in filenames))
        print("Len of additive Noises : ", len(additive_noises))
        rir_noises = []
        for root, _, filenames in os.walk('{}/data/noises/iir/'.format(
                args.datablob)):
            rir_noises.extend((os.path.join(root, name) for name in filenames))

    LOG.info('Found %d additive noise chunks', len(additive_noises))
    LOG.info('Found %d iir chunks', len(rir_noises))

    noise_pipe = td.DataPipe(td.IIDListSampler(additive_noises),
                             td.HDF5Deserializer(),
                             td.Transform(get_hdf5_noise))

    rir_pipe = td.DataPipe(td.IIDListSampler(rir_noises),
                           td.HDF5Deserializer(), td.Transform(get_hdf5_noise),
                           td.RandomizeBuffer(100))

    # synth_block = KwsNoiseSynth('audio_samples',
    #                             'synthaudio',
    #                             noise_pipe,
    #                             rir_pipe,
    #                             snr_samples=[20, 40, 50, 100],
    #                             rir_chance=0.25,
    #                             synth_chance=0.5,
    #                             noise_pooling=50,
    #                             wgn_snr_samples=[20, 40, 50, 100]) ## Previous until 70.pt

    print('************************')
    print('Synth Extra SNR Samples')
    print('************************')
    synth_block = KwsNoiseSynth('audio_samples',
                                'synthaudio',
                                noise_pipe,
                                rir_pipe,
                                snr_samples=[-5, 0, 5, 10, 25, 100, 100],
                                rir_chance=0.25,
                                synth_chance=0.5,
                                noise_pooling=50,
                                wgn_snr_samples=[5, 10, 15, 100, 100])

    # synth_block = KwsNoiseSynth(
    #     'audio_samples',
    #     'synthaudio',
    #     noise_pipe,
    #     rir_pipe,
    #     rir_chance=0.25,
    #     synth_chance=0.5,
    #     noise_pooling=50,
    #     snr_samples=[10],
    #     wgn_snr_samples=[0],
    #     gain_samples=[1.0],
    # )

    return td.DataPipe(
        td.IIDListSampler(chunk_meta,
                          sharding=False,
                          gpu_rank=args.gpu,
                          is_training=args.is_training),
        td.BinaryChunkDeserializer(feat_dim=80),
        td.DecodeWave('audio', 'audio_samples'), td.Filter(_check_16k, 100000),
        td.KeepFields('audio_samples', 'alignment', 'transcription', 'info'),
        synth_block,
        KwsFeatures('synthaudio',
                    table_path=table_path,
                    output_field='feature'), td.Filter(_check_utt_length),
        td.kws.KwsExtractMonophones(phones_file,
                                    'alignment',
                                    output_field='label'),
        td.BasicBatch(sort_batch_size, sort_func=_extract_frame_count),
        td.PrefetchBuffer(),
        KwsBatcher('feature',
                   'label',
                   ignore_label=-100,
                   max_frame_count=batch_frames,
                   list_fields=[
                       'audio_samples', 'info', 'alignment', 'transcription'
                   ]), td.RandomizeBuffer(random_buffer),
        td.ConvertToTensors('feature', 'label', 'seqlen'))