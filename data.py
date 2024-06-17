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


def extract_features(waveform, sample_rate=16000, n_mels=128):
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    return mel_spectrogram.squeeze(0)  # 移除批处理维度


def create_labels(keyword, label_map):
    return [label_map[char] for char in keyword]


class KWSDataset(Dataset):
    def __init__(self, file_paths, labels, label_map, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform = load_and_resample(self.file_paths[idx])
        if self.transform:
            features = self.transform(waveform)
        else:
            features = waveform
        label = create_labels(self.labels[idx], self.label_map)
        input_length = features.shape[-1]
        target_length = len(label)
        return features, torch.tensor(label), input_length, target_length


if __name__ == "__main__":
    # 示例数据
    file_paths = ['path/to/audio1.wav', 'path/to/audio2.wav']
    labels = ['原神启动', '原神启动']
    label_map = {'<blank>': 0, '原': 1, '神': 2, '启': 3, '动': 4}

    dataset = KWSDataset(file_paths, labels, label_map, transform=extract_features)
    features, labels, input_length, target_length = dataset[0]