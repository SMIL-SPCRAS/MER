from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pickle
import torchaudio
import subprocess
import numpy as np
import torch
from transformers import AutoFeatureExtractor
import os
from tqdm import tqdm


def convert_mp4_to_mp3(path, sampling_rate=16000):

    path_save = path[:-3] + "wav"
    if not os.path.exists(path_save):
        ff_audio = "ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}".format(
            path, path_save
        )
        subprocess.call(ff_audio, shell=True)
    wav, sr = torchaudio.load(path_save)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)


def img_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def init(self):
            super(PreprocessInput, self).init()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img, target_size=(224, 224)):
        transform = transforms.Compose([transforms.PILToTensor(), PreprocessInput()])
        img = img.resize(target_size, Image.Resampling.NEAREST)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img

    return get_img_torch(fp)


def pad_sequence(faces, max_length):
    current_length = faces.shape[0]

    if current_length < max_length:
        repetitions = (max_length + current_length - 1) // current_length
        faces = torch.cat([faces] * repetitions, dim=0)[:max_length, ...]

    elif current_length > max_length:
        faces = faces[:max_length, ...]

    return faces


def pad_wav(wav, max_length):
    current_length = len(wav)

    if current_length < max_length:
        repetitions = (max_length + current_length - 1) // current_length
        wav = torch.cat([wav] * repetitions, dim=0)[:max_length]
    elif current_length > max_length:
        wav = wav[:max_length]

    return wav


class AVDataset(Dataset):
    def __init__(
        self,
        path_video="",
        path_images="",
        labels="",
        need_fps=5,
        max_length=2,
        fps_video=24,
        a_step=1,
        corpus="IEMOCAP",
        subset="Train",
        save_path="",
        video=False,
    ):
        self.path_video = path_video
        self.path_images = path_images
        self.labels = labels
        self.sampling_rate = 16000
        self.a_max_length = max_length * self.sampling_rate
        self.a_step = a_step * self.sampling_rate
        self.shape_face = 224
        self.need_fps = need_fps
        self.fps_video = fps_video
        self.step_fps = int(np.round(self.fps_video / self.need_fps))
        self.v_max_length = max_length
        self.zeros = torch.unsqueeze(torch.zeros((3, 224, 224)), 0).cuda()
        self.processor = AutoFeatureExtractor.from_pretrained(
            "DunnBC22/wav2vec2-base-Speech_Emotion_Recognition"
        )
        self.video = video

        self.meta = []

        meta_filename = "{}_max_len_{}_step_{}_av_{}.pickle".format(
            corpus, max_length, a_step, subset
        )

        self.load_data(os.path.join(save_path, meta_filename))

        if not self.meta:
            self.prepare_data()
            self.save_data(os.path.join(save_path, meta_filename))

    def save_data(self, filename):
        with open(filename, "wb") as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as handle:
                self.meta = pickle.load(handle)
        else:
            self.meta = []

    def pth_processing(self, fp):
        class PreprocessInput(torch.nn.Module):
            def init(self):
                super(PreprocessInput, self).init()

            def forward(self, x):
                x = x.to(torch.float32)
                x = torch.flip(x, dims=(0,))
                x[0, :, :] -= 91.4953
                x[1, :, :] -= 103.8827
                x[2, :, :] -= 131.0912
                return x

        def get_img_torch(path):
            img = Image.open(path)
            img = img.resize((224, 224), Image.Resampling.NEAREST)

            ttransform = transforms.Compose(
                [transforms.PILToTensor(), PreprocessInput()]
            )

            img = ttransform(img)
            img = torch.unsqueeze(img, 0).to("cuda")
            return img

        return get_img_torch(fp)

    def prepare_data(self):

        for idx, name_file in enumerate(tqdm(self.path_video)):

            try:
                # print(os.path.join(self.path_images[idx],'00'))
                frames = os.listdir(os.path.join(self.path_images[idx], "00"))
                max_frame = int(frames[-1][:-4]) + 1
                # print(frames)
            except FileNotFoundError:
                max_frame = None

            label = self.labels[idx]
            wav = convert_mp4_to_mp3(name_file, self.sampling_rate)

            for start_a in range(0, len(wav) + 1, self.a_step):
                end_a = min(start_a + self.a_max_length, len(wav))
                if end_a - start_a < self.a_step and start_a != 0:
                    break

                a_fss_chunk = wav[start_a:end_a]
                a_fss = pad_wav(a_fss_chunk, self.a_max_length)
                a_fss = torch.unsqueeze(a_fss, 0)
                a_fss = self.processor(a_fss, sampling_rate=self.sampling_rate)
                a_fss = a_fss["input_values"][0]
                a_fss = torch.from_numpy(a_fss).squeeze(0)

                name_path = []

                start_v = None
                end_v = None

                if max_frame is not None:
                    start_v = int(start_a / self.sampling_rate) * self.fps_video
                    end_v = min(
                        start_v + self.v_max_length * self.fps_video + 1, max_frame
                    )
                    idx_reduction_frames = list(range(start_v, end_v, self.step_fps))

                    for fpath in idx_reduction_frames:
                        fpath = str(fpath).zfill(6)
                        name_path.append(
                            os.path.join(self.path_images[idx], "00", fpath + ".jpg")
                        )

                self.meta.append(
                    {
                        "name_files": name_file,
                        "start_a": start_a,
                        "end_a": end_a,
                        "start_v": start_v,
                        "end_v": end_v,
                        "wav": a_fss,
                        "path_images": name_path,
                        "label": label,
                    }
                )

    def get_meta(self, meta):

        last_frame = False

        v_fss = []

        if self.video:

            paths = meta["path_images"]

            for path in paths:
                if os.path.exists(path):
                    last_frame = True
                    v_fss.append(self.pth_processing(path))
                else:
                    if last_frame:
                        v_fss.append(v_fss[-1])
                    else:
                        v_fss.append(self.zeros)

            if len(v_fss) == 0:
                v_fss.append(self.zeros)

            v_fss = torch.cat(v_fss, dim=0)
            v_fss = pad_sequence(v_fss, self.v_max_length * self.need_fps)

        return meta["name_files"], meta["wav"], v_fss, meta["label"]

    def __getitem__(self, index):
        curr_meta = self.get_meta(self.meta[index])
        audio = torch.FloatTensor(curr_meta[1].to(torch.float))
        if self.video:
            video = torch.FloatTensor(curr_meta[2].to(device="cpu"))
        else:
            video = []
        return curr_meta[0], audio, video, curr_meta[3]

    def __len__(self):
        return len(self.meta)


def get_path_labels_AFEW(subset="Val", dict_emo={}):

    full_path_AFEW = []
    labels = []
    path_images = []

    for emo in dict_emo.keys():
        curr_path = "D:/Databases/AFEW/{}_AFEW/{}/".format(subset, emo)
        curr_path_video = "C:/Work/Faces/IS/AFEW_faces/{}_AFEW/{}/".format(subset, emo)
        curr_name_files = os.listdir(curr_path)
        curr_name_files = [curr_path + i for i in curr_name_files if i.endswith(".wav")]

        path_images.extend(
            [curr_path_video + os.path.basename(i)[:-4] for i in curr_name_files]
        )
        full_path_AFEW.extend(curr_name_files)
        labels.extend([dict_emo[emo]] * len(curr_name_files))
    return full_path_AFEW, path_images, labels


if __name__ == "__main__":

    DICT_EMO_AFEW = {
        "Neutral": 0,
        "Surprise": 3,
        "Fear": 4,
        "Sad": 2,
        "Happy": 1,
        "Disgust": 5,
        "Angry": 6,
    }
    PATH_SAVE_METADATA = "C:/Work/Faces/IS/AFEW_faces/"

    FPS = 24
    MAX_LEN = 4
    STEP = 2

    corpus = "AFEW"
    subset = "Train"
    PATH_VIDEOS, PATH_IMAGES, LABELS = get_path_labels_AFEW(
        subset=subset, dict_emo=DICT_EMO_AFEW
    )

    train_data = AVDataset(
        path_video=PATH_VIDEOS,
        path_images=PATH_IMAGES,
        labels=LABELS,
        a_step=STEP,
        max_length=MAX_LEN,
        fps_video=FPS,
        corpus=corpus,
        subset=subset,
        save_path=PATH_SAVE_METADATA,
        video=True,
    )
