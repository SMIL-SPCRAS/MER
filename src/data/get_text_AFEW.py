import torch
from transformers import pipeline
import os
from tqdm import tqdm
import subprocess
import torchaudio
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_audio_whisper(path, sampling_rate=16000):
    path_save = path[:-3] + "wav"

    if not os.path.exists(path_save):
        subprocess.call(
            f"ffmpeg -i {path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {path_save}",
            shell=True,
        )

    wav, sr = torchaudio.load(path_save)

    wav = wav.mean(dim=0, keepdim=True) if wav.size(0) > 1 else wav
    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)


if __name__ == "__main__":

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        device=device,
    )

    folder = "D:/Databases/AFEW/"

    data = []

    for subset in ["Train_AFEW", "Val_AFEW"]:
        for emo in tqdm(
            ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        ):
            files = [
                i
                for i in os.listdir(os.path.join(folder, subset, emo))
                if i.endswith(".avi")
            ]
            for file in files:
                wav = read_audio_whisper(os.path.join(folder, subset, emo, file))
                text = pipe(wav.numpy(), batch_size=8)["text"]
                data.append((file, subset, emo, text))

    df_AFEW = pd.DataFrame(data, columns=["name_file", "subset", "emo", "text"])

    filename = "df_text_AFEW.csv"
    df_AFEW.to_csv(os.path.join(folder, filename), index=False)
