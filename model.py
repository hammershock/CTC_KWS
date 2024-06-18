import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import make_dataset, extract_features


class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CTCModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 因为是双向LSTM

    def forward(self, x):
        # x: (batch, num_mels, input_seq_len)
        x = x.transpose(1, 2)  # (batch, input_seq_len, num_mels)
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


def train(model, optimizer, dataloader, loss_fn, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, f"Training epoch {epoch}/{num_epochs}"):
            inputs, targets, input_lengths, target_lengths = batch
            # print(input_lengths)
            optimizer.zero_grad()
            outputs = model(inputs)

            # 转换为log_probs以计算CTC损失
            log_probs = F.log_softmax(outputs, dim=2)
            log_probs = log_probs.transpose(0, 1)
            # compute CTC loss
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
        torch.save(model.state_dict(), "./models/model.pt")


def evaluate(model, dataloader, loss_fn):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, input_lengths, target_lengths = batch
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=2)
            log_probs = log_probs.transpose(0, 1)
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    # =============== Model Params ================
    input_dim = 80  # 梅尔频谱图的频带数量, num_mels
    hidden_dim = 256
    output_dim = 5  # 包含所有可能的标签和一个空白标签

    model = CTCModel(input_dim, hidden_dim, output_dim)
    try: model.load_state_dict(torch.load("./models/model.pt"))
    except: pass

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer
    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=0)
    # log_probs: [seq_len, batch_size, num_classes]
    # targets: [batch_size, padded_length]
    # input_lengths: sequence of len(input seqs)
    # target_lengths: sequence of len(target seqs)

    label_map = {" ": 0, "原": 1, "神": 2, "启": 3, "动": 4}
    dataset = make_dataset("./data", "原神启动", label_map=label_map, transform=extract_features)
    features, labels, input_length, target_length = dataset[0]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    train(model, optimizer, dataloader, ctc_loss, num_epochs=10)
    torch.save(model.state_dict(), "./models/model.pt")
    evaluate(model, dataloader, ctc_loss)
