"""Module for keyword spotter components"""

import logging
import math
import io
import itertools
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import scipy.signal
import scipy.io.wavfile
# import openfst_python as openfst
import torch
from torchspeech.utils.memoryviewio import MemoryViewIO
import torchspeech.cpp.kws
# from torchspeech.utils.fstutil import FstUtil
from .datablock import DataBlock
from .spec_augment_pytorch import *

LOG = logging.getLogger(__name__)


class KwsExtractMonophones(DataBlock):
    """
    Data block for extracting monophone label sequence"
    """
    def __init__(self,
                 phones_file,
                 alignment_field,
                 look_ahead=0,
                 length_field=None,
                 legacy_htk=False,
                 output_field='labels'):
        super().__init__()
        self._phones_file = phones_file
        self._alignment_field = alignment_field
        self._length_field = length_field
        self._output_field = output_field
        self._phone_map = None
        self._bad_phones = set()
        self._look_ahead = look_ahead
        self._dropped_utts = 0
        self._total_utts = 0
        self._sr_ms = 10
        self._legacy_htk = legacy_htk
        self._htk_mlf = None

    def reset(self):
        self._dropped_utts = 0
        self._total_utts = 0

        if self._phone_map is None:
            self._phone_map = {}
            phone_list = []
            with open(self._phones_file, 'r') as fhandle:
                for i, line in enumerate(fhandle):
                    phone = line.strip()
                    self._phone_map[phone] = i
                    phone_list.append(phone)

            if self._legacy_htk:
                import torchspeech.cpp.htk
                self._htk_mlf = torchspeech.cpp.htk.mlf(phone_list, 3)

    def status(self):
        """Status of the monophone extraction"""
        if self._dropped_utts > 0:
            count = min(len(self._bad_phones), 10)
            snippet = ', '.join(itertools.islice(self._bad_phones, count))

            return [
                'Total dropped utts: %d' % self._dropped_utts,
                'Bad phones: %dx (%s...)' % (len(self._bad_phones), snippet)
            ]

        return None

    def __next__(self):

        item = self.next_input()
        alignment = item[self._alignment_field]

        if self._legacy_htk:
            phone_seq = self._htk_mlf.process_buffer(alignment)
            phone_seq = (phone_seq - 1).astype(np.int64)

            phone_seq[phone_seq == -1] = -100  # unknown

            end = len(phone_seq)
            cull_start = max(0, end - self._look_ahead)
            phone_seq[cull_start:end] = -100

        else:

            phone_seq = [-100] * int(alignment[-1]['et'] / self._sr_ms)
            self._total_utts += 1
            start = end = 0

            for words in alignment:
                for phone_info in words['phones']:

                    cd_label = phone_info['phone']

                    try:
                        idx0 = cd_label.index('-')
                    except ValueError:
                        idx0 = -1

                    try:
                        idx1 = cd_label.index('+')
                    except ValueError:
                        idx1 = len(cd_label)

                    ci_label = cd_label[idx0 + 1:idx1]

                    try:
                        ci_idx = self._phone_map[ci_label]
                    except KeyError:
                        self._dropped_utts += 1
                        self._bad_phones.add(ci_label)
                        ci_idx = -100  # Mystery phone, give it the ignore index

                    start = int(phone_info['st'] / self._sr_ms)
                    end = int(phone_info['et'] / self._sr_ms)
                    phone_seq[start:end] = (ci_idx for _ in range(start, end))

            cull_start = max(0, end - self._look_ahead)
            phone_seq[cull_start:end] = (-100 for _ in range(cull_start, end))

        item[self._output_field] = phone_seq

        if self._length_field is not None:
            item[self._length_field] = cull_start

        return item


class KwsFeatures(DataBlock):
    """
    A data block for extracting KWS features.
    """
    def __init__(self, sample_field, **kwargs):

        super().__init__()
        self._sample_field = sample_field
        opts = {
            'table_path': None,
            'table': None,
            'output_field': 'feature',
            'max_frames': 10000
        }

        self.set_opts(opts, **kwargs)
        self._handle = None

        if self._table_path is None and self._table is None:
            raise ValueError(
                'If you do not give a binary table, you must give a table path'
            )

        self.zero_feature = self.generate_silence_features()

    def reset(self):
        if self._handle is None:
            if self._table is None:
                with open(self._table_path, 'rb') as fhandle:
                    self._handle = torchspeech.cpp.kws.fe(
                        fhandle.read(), max_frames=self._max_frames)
            else:
                self._handle = torchspeech.cpp.kws.fe(
                    self._table, max_frames=self._max_frames)

    def __next__(self):
        """
        :returns: Next item that meets filter criteria.
        """

        item = self.next_input()
        samples = item[self._sample_field]

        self._handle.reset()
        if samples.dtype == np.int16:
            self._handle.process_16khz_i16(samples)
        elif samples.dtype == np.float32:
            self._handle.process_16khz_float(samples)
        elif samples.dtype == np.float64:
            self._handle.process_16khz_float(samples.astype(np.float32))
        else:
            raise ValueError('Expect samples to be np.int16 or np.float32')

        item[self._output_field] = self._handle.buffer().copy()
        # print(item[self._output_field].shape)
        # count = np.random.randint(0,100)
        # print(count)
        mel_spectrogram = torch.unsqueeze(torch.transpose(torch.from_numpy(item[self._output_field]), 0, 1), 0)
        # self.visualization_spectrogram(mel_spectrogram, "orig", count, 1)
        warped_mel_spectrogram = self.spec_augment(mel_spectrogram)
        # self.visualization_spectrogram(warped_mel_spectrogram, "warped", count, 0)
        item[self._output_field] = (torch.transpose(torch.squeeze(warped_mel_spectrogram), 0, 1)).cpu().detach().numpy()
        return item

    def spec_augment(self, mel_spectrogram):
        warped_masked_spectrogram = spec_augment(mel_spectrogram=mel_spectrogram, time_warping_para=5, frequency_masking_para=15,
                 time_masking_para=20, frequency_mask_num=2, time_mask_num=2)
        # print("done")
        return warped_masked_spectrogram

    def visualization_spectrogram(self, mel_spectrogram, title, num, orig=1):
        """visualizing result of SpecAugment
        # Arguments:
        mel_spectrogram(ndarray): mel_spectrogram to visualize.
        title(String): plot figure's title
        """
        # Show mel-spectrogram using librosa's specshow.
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram[0, :, :])
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()
        if orig==1:
            plt.savefig('/home/t-sagoy/samples/orig{}.png'.format(num))
            # print("done")
        else:
            plt.savefig('/home/t-sagoy/samples/warp{}.png'.format(num))

    def generate_silence_features(self):
        samples = np.zeros(1000, dtype=np.float32)
        if self._handle is None:
            self.reset()

        self._handle.process_16khz_float(samples)
        features = self._handle.buffer().copy()

        return features[0][0]

    def get_silence_features(self):
        return self.zero_feature


class KwsAmEngine(KwsFeatures):
    """
    Module for extracting KWS table features.
    """
    def __init__(self, sample_field, **kwargs):
        super().__init__(sample_field, **kwargs)

    def reset(self):
        if self._handle is None:
            if self._table is None:
                with open(self._table_path, 'rb') as fhandle:
                    self._handle = torchspeech.cpp.kws.am(
                        fhandle.read(), max_frames=self._max_frames)
            else:
                self._handle = torchspeech.cpp.kws.am(
                    self._table, max_frames=self._max_frames)


class KwsNoiseSynth(DataBlock):
    """
    A simple noisy file synthesis: y = h*x + n

    :param audio_field: audio field name
    :type audio_field: string
    :param output_field: output field to store synthezied audio
    :type output_field: string
    :param noise_source: source of noise samples
    :type noise_source: iterable
    :param rir_source: source of rir samples
    :type rir_source: iterable

    Optional kwags:

    :param snr_samples: list of snr values to sample from
    :type snr_samples: list<float>
    :param gain_samples: list of gain values to sample from
    :type gain_samples: list<float>
    :param rir_chance: chance to use an rir [0..1]
    :type rir_chance: float
    :param synth_chance: probability to do any synthesis [0..1]
    type synth_chance: float
    :param noise_pooling: size of noise randomization pool
    :type noise_pooling: 50
    """
    def __init__(self, audio_field, output_field, noise_source, rir_source,
                 **kwargs):

        super().__init__()
        self._audio_field = audio_field
        self._output_field = output_field
        self._noise_source = iter(noise_source)
        self._rir_source = iter(rir_source)

        self._noise_list = None
        self._rir_list = None

        opt_map = {
            'snr_samples': [-10, -10, 0, 0, 10, 20],
            'wgn_snr_samples': [-10, -10, 0, 0, 10, 20],
            'gain_samples': [1.0, 0.25, 0.5, 0.75],
            'rir_chance': 0.75,
            'synth_chance': 0.8,
            'noise_pooling': 50,
            'max_samples': 160000,
            'min_samples': 32000
        }

        self.set_opts(opt_map, **kwargs)

    def reset(self):
        if self._noise_list is None:
            self._noise_list = []
            for kk in range(self._noise_pooling):
                # print("noise pooling number : ", kk)
                # print("Len of Noise list : ", len(self._noise_list))
                self._noise_list.append(self.__next_raw_noise())

    def __next__(self):
        for _ in range(100):
            item = self.next_input()
            audio_field = item[self._audio_field]
            samples = self.__extract_16k_float_audio(audio_field)
            # print(samples)
            if samples is None or len(samples) < 32000:
                continue

            if np.random.random() < self._synth_chance:

                sigy = self.__synthesize_wave(
                    samples, np.random.choice(self._snr_samples),
                    np.random.choice(self._wgn_snr_samples),
                    np.random.choice(self._gain_samples),
                    np.random.random() < self._rir_chance)

                if sigy is not None:
                    item[self._output_field] = sigy
                    return item

            else:
                item[self._output_field] = samples
                return item

        raise ValueError('Failed to produce a record (tried 10 times)')

    def __extract_16k_float_audio(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.int16 or obj.dtype == np.float32 or obj.dtype == np.float64:
                samples = obj
                sample_rate = 16000
            else:
                print("*****")
                print(obj)
                print(type(obj))
                print(obj.dtype)
                print("****")
                # print(obj.data[0].dtype())
                # import pdb;pdb.set_trace()
                memv = MemoryViewIO(obj.data)
                sample_rate, samples = scipy.io.wavfile.read(memv)
        else:
            # print("loops 2")
            try:
                with io.BytesIO(obj) as memv:
                    sample_rate, samples = scipy.io.wavfile.read(memv)
            except:
                print("couldn read")
                return None

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) * 0.000030517578125

        if sample_rate != 16000:
            print("*******samp rate was not 16k!*****")
            return None

        return samples

    def __next_raw_noise(self):
        try:
            # print("Noise Source")
            # print(self._noise_source)
            noise_wav = next(self._noise_source)
            # print("printing the object")
            # print(noise_wav)
        except StopIteration:
            raise ValueError('The noise iterator ran out of data!')

        noise_samples = self.__extract_16k_float_audio(noise_wav)
        # x = np.random.randint(0,1000)
        # sf.write(f'audio{x}.wav', noise_samples, 16000)
        rmse = math.sqrt(
            np.sum(noise_samples * noise_samples) / len(noise_samples))
        return noise_samples, rmse

    def __get_random_noise_slice(self, num_samples):
        # print("len of noise list at rand samp : ", len(self._noise_list))
        idx = np.random.random_integers(len(self._noise_list)) - 1
        samples, rmse = self._noise_list[idx]

        if len(samples) < num_samples:

            # Not enough samples left, so this will get only a patch of noise
            self._noise_list[idx] = self.__next_raw_noise()
            padded = np.zeros((num_samples), dtype=np.float32)
            padded[0:len(samples)] = samples
            return padded, rmse

        else:
            sample_cut = samples[0:num_samples]

            # If there is >3s left in the noise, put the trimmed
            # portion back, otherwise load in a new sample
            if len(samples) - num_samples >= 48000:
                # print("it is fine : ", len(samples))
                self._noise_list[idx] = (samples[num_samples:], rmse)
            else:
                # print("Lets us sample next ")
                self._noise_list[idx] = self.__next_raw_noise()
            return sample_cut, rmse

    def __synthesize_wave(self, sigx, snr, wgn_snr, gain, do_rir):

        if len(sigx) < self._min_samples:
            return None

        if len(sigx) > self._max_samples:
            sigx = sigx[0:self._max_samples]
        
        beta = np.random.choice([0.1, 0.25, 0.5, 0.75, 1])
        sigx = beta * sigx
        x_power = np.sum(sigx * sigx)

        # Do RIR and normalize back to original power
        if do_rir:
            try:
                next_rir = next(self._rir_source)
            except StopIteration:
                raise ValueError('The RIR iterator given has stopped!')

            rir_sample = self.__extract_16k_float_audio(next_rir)

            # We cut the tail of the RIR signal at 99% energy
            cum_en = np.cumsum(np.power(rir_sample, 2))
            cum_en = cum_en / cum_en[-1]
            rir_sample = rir_sample[cum_en <= 0.99]

            max_spike = np.argmax(np.abs(rir_sample))
            sigy = scipy.signal.fftconvolve(sigx, rir_sample)[max_spike:]
            sigy = sigy[0:len(sigx)]

            y_power = np.sum(sigy * sigy)
            sigy *= math.sqrt(x_power /
                              y_power)  # normalize so y has same total power

        else:
            sigy = sigx

        y_rmse = math.sqrt(x_power / len(sigy))

        # Only bother with noise addition if the SNR is low enough
        if snr < 50:
            noise_samps, noise_rmse = self.__get_random_noise_slice(len(sigy))
            noise_scale = y_rmse / noise_rmse * math.pow(10, -snr / 20)
            sigy = sigy + noise_samps * noise_scale

        if wgn_snr < 50:
            wgn_samps = np.random.normal(size=(len(sigy))).astype(np.float32)
            noise_scale = y_rmse * math.pow(10, -wgn_snr / 20)
            sigy = sigy + wgn_samps * noise_scale

        # Apply gain & clipping
        return np.clip(sigy * gain, -1.0, 1.0)


class KwsPhoneFST(DataBlock):
    """
    A data block for preparing phone alignment FSTs.
    """
    def __init__(self,
                 lexicon_file,
                 phone_file,
                 silence_label,
                 transcription_field,
                 output_field='phone_fst'):
        super().__init__()

        self._lexicon_file = lexicon_file
        self._phone_file = phone_file
        self._transcription_field = transcription_field
        self._output_field = output_field
        self._silence_label = silence_label
        self._word_map = None
        self._fst_cache = {}
        self._phone_symbols = None
        self._sil_index = -1
        self._sil_fst = None
        self._oov = set()
        self._total_utts = 0
        self._failed_utts = 0

    def status(self):
        """Status of the phone FST block"""
        if self._failed_utts > 0:
            snippet = ', '.join(itertools.islice(self._oov, 20))
            arg_fmt = '%d / %d utts failed'

            return [arg_fmt % (self._failed_utts, self._total_utts), snippet]

    def reset(self):

        self._failed_utts = 0
        self._total_utts = 0

        if self._word_map is not None:
            return

        self._word_map = {}

        with open(self._phone_file) as phonef:
            phone_list = [x.strip() for x in phonef]
            self._phone_symbols = FstUtil.create_symbol_table(phone_list)

        self._sil_index = self._phone_symbols.find(self._silence_label)
        if self._sil_index <= 0:
            raise ValueError(
                'Cannot find silence label "%s" or it is mapped to 0 (for eps)'
                % self._silence_label)

        self._sil_fst = FstUtil.create_self_loop_fst(self._sil_index)

        with open(self._lexicon_file, 'r') as lexf:
            for line in lexf:
                fields = line.strip().lower().split(' ')
                word = fields[0]
                phones = fields[1:]

                self._word_map.setdefault(word, []).append(phones)

    def __next__(self):
        for _ in range(30):
            item = self.next_input()
            self._total_utts += 1
            transcript = item[self._transcription_field]
            words = transcript.strip().lower().split()
            fst = self.__create_utterance_fst(words)

            if fst is not None:
                item[self._output_field] = fst
                return item

            self._failed_utts += 1

        raise ValueError('Failed to generate FST but failed with OOVs (30x)')

    def __create_word_fst(self, word):

        try:
            return self._fst_cache[word]
        except KeyError:
            pass

        try:
            label_sequences = self._word_map[word]
        except KeyError:
            self._fst_cache[word] = None
            self._oov.add(word)
            return None

        word_fst = FstUtil.fst_from_label_sequence(
            self._phone_symbols,
            label_sequences,
            exit_pad_index=self._sil_index)

        self._fst_cache[word] = word_fst
        return word_fst

    def __create_utterance_fst(self, words, mindet=False):
        """
        Create an alignment FST using a sequence of words.
        """
        word_fsts = (self.__create_word_fst(word) for word in words)
        utt_fst = FstUtil.concatenate_fsts(self._sil_fst, *word_fsts)

        if utt_fst is None:
            self._failed_utts += 1
            return None

        if mindet:
            utt_fst = openfst.determinize(utt_fst)
            utt_fst = utt_fst.minimize()

        return openfst.epsnormalize(utt_fst)


class KwsBatcher(DataBlock):
    """
    Data block for batching KWS data
    """
    def __init__(self, feature_field, label_field, **kwargs):

        super().__init__()

        opts = {
            'ignore_label': -100,
            'max_frame_count': 4096,
            'pre_batch_size': 5000,
            'lattice_field': None,
            'list_fields': [],
            'use_single_batch': False,
            'pad_value': 0.0,
            'silence_feature': 0.0
        }

        self.set_opts(opts, **kwargs)

        self._feature_field = feature_field
        self._label_field = label_field

        self._current_batch = iter([])
        self._toggle = False
        self._previous_sample = None
        self._fetch_iter = None

    def __get_frame_count(self, item):
        return item[self._feature_field].shape[0]

    def __get_non_silence_frame_count(self, item):

        length = item[self._feature_field].shape[0]
        channels = item[self._feature_field].shape[1]
        new_length = length
        for i in range(length - 1, 0, -1):
            if np.all(item[self._feature_field][i] == np.full(
                (channels), self._silence_feature, dtype=np.float32)):
                new_length -= 1
            else:
                break
        if new_length <= 20:
            print(
                f'found sample with {new_length} new length and actual length {length}'
            )
            new_length = length
        return new_length

    def reset(self):
        self._previous_sample = None

    def __next_sample(self):
        try:
            return next(self._current_batch)
        except StopIteration:
            pass

        new_batch = self.next_input()

        if self._toggle:
            self._toggle = False
            self._current_batch = reversed(new_batch)
        else:
            self._toggle = True
            self._current_batch = iter(new_batch)

        return next(self._current_batch)

    def __next__(self):

        if self._use_single_batch:
            # return self.__pad_batch(self.next_input())

            if self._previous_sample is None:
                self._previous_sample = self.__next_sample()

            batch = [self._previous_sample]
            current_batches = 1
            while True:
                try:
                    item = self.__next_sample()

                    if current_batches >= 1:
                        self._previous_sample = item
                        return self.__pad_batch(batch)
                    else:
                        batch.append(item)
                        current_batches += 1
                except StopIteration:
                    self._previous_sample = None
                    return self.__pad_batch(batch)

        else:

            if self._previous_sample is None:
                self._previous_sample = self.__next_sample()

            batch = [self._previous_sample]
            current_frames = self.__get_frame_count(batch[0])
            while True:
                try:
                    item = self.__next_sample()
                    new_count = current_frames + self.__get_frame_count(item)

                    if new_count > self._max_frame_count:
                        self._previous_sample = item
                        return self.__pad_batch(batch)
                    else:
                        batch.append(item)
                        current_frames = new_count
                except StopIteration:
                    self._previous_sample = None
                    return self.__pad_batch(batch)

    def __pad_batch(self, batch):

        batch_sz = len(batch)
        channels = batch[0][self._feature_field].shape[1]
        output_item = {}

        feat_sz = [self.__get_frame_count(item) for item in batch]
        feat_sz_non_silence = [
            self.__get_non_silence_frame_count(item) for item in batch
        ]
        if self._label_field is None:
            frame_count = feat_sz
        else:
            lab_sz = [len(item[self._label_field]) for item in batch]
            frame_count = [min(feat_sz[i], lab_sz[i]) for i in range(batch_sz)]

        max_frames = max(frame_count)

        feature_data = np.full((len(batch), max_frames, channels),
                               self._pad_value,
                               dtype=np.float32)
        output_item[self._feature_field] = feature_data

        for i, item in enumerate(batch):
            data = item[self._feature_field][0:frame_count[i], :]
            feature_data[i, 0:data.shape[0], :] = data

        if self._label_field is not None:
            label_data = np.full((len(batch), max_frames),
                                 self._ignore_label,
                                 dtype=np.int64)
            output_item[self._label_field] = label_data

            for i, item in enumerate(batch):
                data = item[self._label_field][0:frame_count[i]]
                label_data[i, 0:frame_count[i]] = data

        for field in self._list_fields:
            output_item[field] = [item[field] for item in batch]

        output_item['frames'] = sum(frame_count)
        output_item['seqlen'] = np.array(feat_sz, dtype=np.int32)
        output_item['actual_seqlen'] = np.array(feat_sz_non_silence,
                                                dtype=np.int32)

        if self._lattice_field is not None:
            lattices = [item[self._lattice_field] for item in batch]
            pack = FstUtil.pack_batch(lattices)
            fstinfo = torchspeech.cpp.fst.get_fstinfo(pack.tobytes(),
                                                      output_item['seqlen'])
            output_item[self._lattice_field] = pack
            output_item['fstinfo'] = fstinfo

        return output_item
