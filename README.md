# CTC-KWS

基于CTC的关键字唤醒（KWS）模型实现

## 1. KWS 概述

随着人工智能的飞速发展，市场上推出了各式各样的智能设备，AI 语音的发展更是使得语音助手成为各大智能终端设备必不可少的软件。语音是人类与设备最直接的交互方式，不需要和实物接触，可远程操控，对于人们来说是最方便自然的交流方式。

自动语音识别（Automatic Speech Recognition, ASR）是一种将语音转化为文字的技术，是人与机器、人与人自然交流的关键技术之一。ASR 是人与智能设备交互的入口，它的功能就是让设备“听懂”人类的语言，从而能够根据识别到的内容去完成人类想要让它做的事情。

语音唤醒（Keyword Spotting, KWS）是语音识别的入口，如何高效、准确地对用户指令给出反应成为这一技术的最重要的目标。

下图是 IPhone 中 Siri 语音助手的交互示意图，总体上可分为以下三个步骤： 
1. 麦克风持续检测声音信号 
2. 逐帧对声音信号进行特征提取和模型预测 
3. 当接收到一个完整的 "Hey Siri" 的语音时，此刻模型的得分达到最大值，触发唤醒事件

图片来源：https://machinelearning.apple.com/research/hey-siri

## 1.1 产品应用

Apple 广告中 Siri 语音助手的交互演示视频。

## 1.2 KWS、ASR 和声音检测

KWS、ASR 和声音检测的关系： 
- KWS VS ASR：KWS 可以看作是一类特殊的 ASR，它只识别声音中的固定的关键词。ASR 需要语言模型来理解一段声音中的文字，而 KWS 仅需关注固定样式的发音。从模型输入输出的角度看，KWS 输入音频，输出是判别结果；ASR 输入音频，输出是文字序列。
- KWS VS 声音检测：KWS 和声音检测都是捕获特定的声音，KWS 注重语音中的关键词，而声音检测的范围更为宽泛，可以是自然界中的语音，工业领域里机器产生的声音，人类的哭声，尖叫声等异常声音。从模型输入输出的角度看，KWS 和声音检测都是输入音频，输出判别结果。

图片来源：http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/Speaker%20(v3).pdf

## 2. 适用于 KWS 的模型

### 2.1 传统 HMM 模型

与语音识别 ASR 类似，KWS 可以用传统的 HMM 模型完成建模和识别，模型结构上也是声学模型加解码器。

基于 HMM 的 KWS 模型和传统 ASR 模型的区别： 
- 声学模型：KWS 只需关注少量的音素，对于其他发音可以当作 Filler 处理，因此声学模型的类别数可以做到很低，譬如在单音素建模下只需要 10 个以内；而 ASR 面向所有的发音，音素全，因此声学模型类别数会大很多。
- 解码器：ASR 的解码器是一个 FST，输入声学模型的结果输出文字序列；而 KWS 的解码器是一个 FSA，如果到达最终状态可以给出一个得分作为唤醒词的分数，解码图的大小相对于 ASR 会小很多。

图片来源：https://developer.nvidia.com/blog/how-to-build-domain-specific-automatic-speech-recognition-models-on-gpus

### 2.2 端到端模型

#### 2.2.1 基于后验概率平滑的模型

在 2014 年的文章 Small-footprint keyword spotting using deep neural networks 中，作者提出了一种基于神经网络加后验概率平滑的 KWS 方法，该方法利用词粒度来建模声学模型，可分为以下四个执行步骤： 
- 特征提取和重叠：对音频信号进行逐帧的特征提取，预测每一帧的声学概率分布时，加入了上下文信息（前30帧+后10帧）后作为模型输入。
- 声学模型：对叠加了上下文信息的频域特征进行声学概率分布的预测，模型总共有 N 个标签，其中标签 0 为 Filler，将与唤醒词的发音无关的归类至此。
- 后验概率平滑：从声学模型得到整段音频的声学概率分布后，采用滑窗的进行后验概率平滑的计算，这么做可以去除一些噪音，增强鲁棒性。
- 唤醒词得分计算：引入一个得分窗口，在窗口内统计除 Filler 外的所有声学概率的最大值，通过累乘和开方的计算方式得到最终得分。

图片来源：https://ieeexplore.ieee.org/document/6854370

#### 2.2.2 基于 Max-Pooling Loss 的模型

在 2017 年的文章 Max-Pooling Loss Training of Long Short-Term Memory Networks for Small-Footprint Keyword Spotting 中，作者提出了一种基于 Max-Pooling Loss 的 KWS 模型训练方法。

这种方法可以看作是从帧级别的训练方式转向段级别的训练方式，如下图所示，蓝色填充的帧是唤醒词，在训练阶段，模型对于唤醒词片段的得分取决于某一帧中的最高的得分；而非唤醒词片段，为了保证所有帧的得分都足够低，则需要关注所有的帧。这种得分可以看作是基于声学得分的 Max-Pooling。

有了这个训练方式，我们直接地对唤醒词进行端到端的建模，具体模型可以采取 RNN-based、CNN-based 和 Attention-based 可对音频特征序列建模的模型。PaddleSpeech 中的 examples/hey_snips 采用了 Multi-scale Dilated Temporal Convolutional 模型，通过 Max-Pooling Loss 的训练方法实现了在 Snips 数据集上的训练和评估。

图片来源：https://arxiv.org/pdf/1705.02411.pdf

## 3. 实践：KWS模型训练和评估

本节将展示如何使用PyTorch实现一个CTC模型用于关键字唤醒任务，并进行训练和评估。

### 3.1 环境准备

首先，确保已经安装了PyTorch和必要的依赖包：

```bash
pip install torch torchaudio
```
