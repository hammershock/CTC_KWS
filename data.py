import glob

import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram
import torch
from torch.utils.data import Dataset, DataLoader


def load_and_resample(file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def extract_features(waveform, sample_rate=16000, n_mels=80):
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    return mel_spectrogram.squeeze(0)  # 移除批处理维度


def create_labels(keyword, label_map):
    return [label_map[char] for char in keyword]


class KWSDataset(Dataset):
    def __init__(self, file_paths, labels, label_map, transform=lambda x: x):
        self.file_paths = file_paths
        self.labels = labels
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform = load_and_resample(self.file_paths[idx])
        features = self.transform(waveform)
        label = create_labels(self.labels[idx], self.label_map)
        input_length = features.shape[-1]
        target_length = len(label)
        return features, torch.tensor(label), input_length, target_length


def make_dataset(dir_path, keyword, label_map, transform=None):
    positive_files = glob.glob(f"{dir_path}/pos/*.wav", recursive=True)
    negative_files = glob.glob(f"{dir_path}/neg/*.wav", recursive=True)
    filepaths = positive_files + negative_files
    labels = [keyword] * len(positive_files) + [" "] * len(negative_files)
    return KWSDataset(filepaths, labels, label_map, transform)


if __name__ == "__main__":
    label_map = {" ": 0, "原": 1, "神": 2, "启": 3, "动": 4}
    dataset = make_dataset("./data", "原神启动", label_map=label_map, transform=extract_features)
