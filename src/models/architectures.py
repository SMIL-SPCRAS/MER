from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from transformers import AutoConfig

import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import requests


class AudioModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

    def extract_features(self, x):
        outputs = self.wav2vec2(x)
        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        return hidden_states

    def forward(self, x):
        hidden_states = self.extract_features(x)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same", bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.batch_norm3 = nn.BatchNorm2d(
            out_channels * self.expansion, eps=0.001, momentum=0.99
        )

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv_layer_s2_same = Conv2dSame(
            num_channels, 64, 7, stride=2, groups=1, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * ResBlock.expansion, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def extract_features(self, x):
        x = self.relu(self.batch_norm1(self.conv_layer_s2_same(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    padding=0,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion, eps=0.001, momentum=0.99),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


class GFL(nn.Module):
    def __init__(self, input_dim_F1, input_dim_F2, gated_dim):
        super(GFL, self).__init__()

        self.WF1 = nn.Parameter(torch.Tensor(input_dim_F1, gated_dim))
        self.WF2 = nn.Parameter(torch.Tensor(input_dim_F2, gated_dim))

        init.xavier_uniform_(self.WF1)
        init.xavier_uniform_(self.WF2)

        dim_size_f = input_dim_F1 + input_dim_F2

        self.WF = nn.Parameter(torch.Tensor(dim_size_f, gated_dim))

        init.xavier_uniform_(self.WF)

    def forward(self, f1, f2):

        h_f1 = torch.tanh(torch.matmul(f1, self.WF1))
        h_f2 = torch.tanh(torch.matmul(f2, self.WF2))
        z_f = torch.softmax(torch.matmul(torch.cat([f1, f2], dim=1), self.WF), dim=1)
        h_f = z_f * h_f1 + (1 - z_f) * h_f2
        return h_f


class AVTmodel(nn.Module):
    def __init__(
        self,
        input_dim_audio,
        input_dim_video,
        input_dim_text,
        gated_dim,
        n_classes,
        drop,
    ):
        super(AVTmodel, self).__init__()

        self.fc_audio = nn.Linear(input_dim_audio, input_dim_audio)
        self.drop_audio = nn.Dropout(p=drop)
        self.fc_video = nn.Linear(input_dim_video, input_dim_video)
        self.drop_video = nn.Dropout(p=drop)
        self.fc_text = nn.Linear(input_dim_text, input_dim_text)
        self.drop_text = nn.Dropout(p=drop)

        init.xavier_uniform_(self.fc_audio.weight)
        init.xavier_uniform_(self.fc_video.weight)
        init.xavier_uniform_(self.fc_text.weight)

        self.gal_av = GFL(input_dim_audio, input_dim_video, gated_dim)
        self.gal_vt = GFL(input_dim_video, input_dim_text, gated_dim)
        self.gal_ta = GFL(input_dim_text, input_dim_audio, gated_dim)

        self.fc_avt = nn.Linear(gated_dim * 3, gated_dim)
        self.drop_avt = nn.Dropout(p=drop)
        self.classifier = nn.Linear(gated_dim, n_classes)

        init.xavier_uniform_(self.fc_avt.weight)
        init.xavier_uniform_(self.classifier.weight)

    def forward(self, audio, video, text):
        fc_audio = self.drop_audio(self.fc_audio(audio))
        fc_video = self.drop_video(self.fc_video(video))
        fc_text = self.drop_text(self.fc_text(text))

        h_av = self.gal_av(fc_audio, fc_video)
        h_vt = self.gal_vt(fc_video, fc_text)
        h_ta = self.gal_ta(fc_text, fc_audio)

        h_avt = torch.cat([h_av, h_vt, h_ta], dim=1)

        h_avt = nn.functional.relu(h_avt)
        h_avt = self.drop_avt(self.fc_avt(h_avt))
        prob = self.classifier(h_avt)
        return prob


def load_model(model_url, model_path):
    try:
        with requests.get(model_url, stream=True) as response:
            with open(model_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        return model_path
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # audio model initialization
    path_audio_model = "DunnBC22/wav2vec2-base-Speech_Emotion_Recognition"
    sampling_rate = 16000

    config = AutoConfig.from_pretrained(path_audio_model)
    num_classes = 7
    config.num_labels = num_classes

    audio_model = AudioModel.from_pretrained(
        path_audio_model, config=config, ignore_mismatched_sizes=True
    )
    audio_model.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
    audio_model.to(device).eval()

    inp_audio = torch.zeros((4, sampling_rate * 4))

    res = audio_model(inp_audio.to(device))
    print("audio_outputs", res)

    # video model initialization
    model_url = "FER_static_ResNet50_AffectNet.pt"
    model_name = "FER_static_ResNet50_AffectNet.pt"

    path_video_path = load_model(model_url, model_name)

    video_model = ResNet50(num_classes=7, channels=3)
    video_model.load_state_dict(torch.load(path_video_path))
    video_model.to(device).eval()

    inp_video = torch.zeros((4, 3, 224, 224))

    res = video_model(inp_video.to(device))
    print("video_outputs", res)

    # multimodal model initialization
    avt_model = AVTmodel(512, 1024, 768, gated_dim=64, n_classes=7, drop=0.2)
    avt_model.to(device).eval()

    inp_audio = torch.zeros((4, 512))
    inp_video = torch.zeros((4, 1024))
    inp_text = torch.zeros((4, 768))
    res = avt_model(inp_audio.to(device), inp_video.to(device), inp_text.to(device))
    print("avt_outputs", res)
