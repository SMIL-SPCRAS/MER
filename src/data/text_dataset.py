import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.tokenized_texts = tokenized_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_texts["input_ids"][idx],
            "attention_mask": self.tokenized_texts["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx]),
        }


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base"
    )

    path_df = "D:/Databases/AFEW/df_text_AFEW.csv"

    emo_AFEW = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    df = pd.read_csv(path_df)

    emo_train = df[(df.subset == "Train_AFEW")].emo.tolist()
    emo_train = [emo_AFEW.index(i) for i in emo_train]
    text_train = df[df.subset == "Train_AFEW"].text.tolist()

    tokenized_texts_train = tokenizer(
        text_train, padding=True, truncation=True, return_tensors="pt"
    )

    data_train = TextDataset(tokenized_texts=tokenized_texts_train, labels=emo_train)
