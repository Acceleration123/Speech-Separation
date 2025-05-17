"""
code to create mixture dataset for speech separation task
fs: 16000Hz
length for each mixture: 5s
number of speakers: 2
no reverberation & no noise added
"""
import librosa
import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm

T = 5  # length for each mixture in seconds


def mk_mix(spk1, spk2, fs, snr1, snr2):
    n_samples = int(fs * T)

    if len(spk1) < n_samples:
        spk1 = np.pad(spk1, (0, n_samples - len(spk1)))
    else:
        spk1 = spk1[:n_samples]

    if len(spk2) < n_samples:
        spk2 = np.pad(spk2, (0, n_samples - len(spk2)))
    else:
        spk2 = spk2[:n_samples]

    scale_factor1 = 10 ** (snr1 / 20)
    scale_factor2 = 10 ** (snr2 / 20)

    return spk1, spk2, scale_factor1 * spk1 + scale_factor2 * spk2


def mk_mix_dataset(data_dir, dataset_type, output_dir):
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)

    spk1_dir = os.path.join(dataset_dir, "spk1")
    spk2_dir = os.path.join(dataset_dir, "spk2")
    mix_dir = os.path.join(dataset_dir, "mix")

    os.makedirs(spk1_dir, exist_ok=True)
    os.makedirs(spk2_dir, exist_ok=True)
    os.makedirs(mix_dir, exist_ok=True)

    file_length = 5
    speech_files = sorted(librosa.util.find_files(data_dir))
    for spk1_idx in tqdm(range(len(speech_files) - 1)):
        spk2_idx = np.random.randint(spk1_idx + 1, len(speech_files))

        spk1, fs = sf.read(speech_files[spk1_idx], dtype='float32')  # fs: 16000Hz
        spk2, fs = sf.read(speech_files[spk2_idx], dtype='float32')  # fs: 16000Hz

        snr1 = np.random.uniform(low=0, high=0.5)
        snr2 = snr1 + np.random.uniform(low=-0.5, high=0.5)
        spk1_processed, spk2_processed, mix = mk_mix(spk1, spk2, fs, snr1, snr2)

        sf.write(os.path.join(spk1_dir, f"{spk1_idx:0{file_length}d}.wav"), spk1_processed, fs)
        sf.write(os.path.join(spk2_dir, f"{spk1_idx:0{file_length}d}.wav"), spk2_processed, fs)
        sf.write(os.path.join(mix_dir, f"{spk1_idx:0{file_length}d}.wav"), mix, fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-tr', '--train_database', type=str,
        help='Path to train speech database',
        default="D:\\Users\\14979\\Desktop\\LibriSpeech_train_clean\\train-clean-360"
    )

    parser.add_argument(
        '-va', '--valid_database', type=str,
        help='Path to validation speech database',
        default="D:\\Users\\14979\\Desktop\\LibriSpeech_dev_clean\\dev-clean"
    )

    parser.add_argument(
        '-te', '--test_database', type=str,
        help='Path to test speech database',
        default="D:\\Users\\14979\\Desktop\\LibriSpeech_test_clean\\test-clean"
    )

    parser.add_argument(
        '-o', '--output_dir', type=str,
        help='Path to output directory',
        default="D:\\Users\\LibriMix_Mine"
    )

    args = parser.parse_args()

    mk_mix_dataset(args.train_database, "train", args.output_dir)
    mk_mix_dataset(args.valid_database, "valid", args.output_dir)
    mk_mix_dataset(args.test_database, "test", args.output_dir)


