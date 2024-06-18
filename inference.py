from collections import deque

import sounddevice as sd
import queue
import threading
import torch
import torchaudio
import numpy as np
from torchaudio.transforms import MelSpectrogram
from model import CTCModel  # 假设模型定义在 model.py 文件中

# Configuration
sample_rate = 16000
n_mels = 80
buffer_size = 2048
hop_length = 512
window_size = 64

# Queue to hold audio data
q = queue.Queue()

# Load the trained model
input_dim = 80  # 梅尔频谱图的频带数量, num_mels
hidden_dim = 256
output_dim = 5  # 包含所有可能的标签和一个空白标签

model = CTCModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("./models/model.pt"))
model.eval()

# Label map for decoding
label_map = {0: " ", 1: "原", 2: "神", 3: "启", 4: "动"}
inverse_label_map = {v: k for k, v in label_map.items()}

# MelSpectrogram transform
mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length)


def extract_features(waveform):
    mel_spec = mel_spectrogram(waveform)
    return mel_spec.squeeze(0)  # 移除批处理维度


# Function to process audio and detect keywords
def detection_loop():
    print("Listening ...")
    buffer = deque(maxlen=32000 * 5)
    while True:
        if not q.empty():
            data = q.get()  # [chunk_size]
            buffer.extend(data)
            waveform = torch.tensor(np.array(buffer), dtype=torch.float32)
            features = extract_features(waveform)
            features = features.unsqueeze(0)  # [batch, num_mels, mel_len]

            with torch.no_grad():
                # print(features.shape)
                outputs = model(features)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
                pred = torch.argmax(log_probs, dim=2).squeeze(0)
                print(pred)
                # Decode the prediction
                pred_str = ''.join([label_map[c.item()] for c in pred if c.item() in label_map])
                # print(f"Detected sequence: {pred_str}")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    data = indata[:, 0]
    q.put(data)


# Start a new thread for detection loop
detect_thread = threading.Thread(target=detection_loop)
detect_thread.daemon = True
detect_thread.start()

duration = 100  # Duration to record in seconds

# Start audio stream
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=buffer_size):
    sd.sleep(duration * 1000)  # Record for the given duration
