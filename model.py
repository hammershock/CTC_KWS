import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import labels, extract_features, KWSDataset


class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CTCModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 因为是双向LSTM

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def collate_fn(batch):
    features, labels, input_lengths, target_lengths = zip(*batch)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = torch.tensor(input_lengths)
    target_lengths = torch.tensor(target_lengths)
    return features, labels, input_lengths, target_lengths


def add_noise(waveform, noise_factor=0.005):
    noise = torch.randn(waveform.size()) * noise_factor
    augmented_waveform = waveform + noise
    return augmented_waveform


def train(model, optimizer, dataloader, num_epochs):
    model.train()

    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=0)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, targets, input_lengths, target_lengths = batch

            optimizer.zero_grad()
            outputs = model(inputs)

            # 转换为log_probs以计算CTC损失
            log_probs = F.log_softmax(outputs, dim=2)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


def evaluate(model, dataloader):
    model.eval()

    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=0)

    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, input_lengths, target_lengths = batch
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=2)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    # =============== Model Params ================
    input_dim = 128  # 梅尔频谱图的频带数量
    hidden_dim = 256
    output_dim = len(labels) + 1  # 包含所有可能的标签和一个空白标签

    model = CTCModel(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

    # 示例数据
    file_paths = ['path/to/audio1.wav', 'path/to/audio2.wav']
    labels = ['原神启动', '原神启动']
    label_map = {'<blank>': 0, '原': 1, '神': 2, '启': 3, '动': 4}

    dataset = KWSDataset(file_paths, labels, label_map, transform=extract_features)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    train(model, optimizer, dataloader, num_epochs=10)

    # evaluate(model, validation_dataloader)
