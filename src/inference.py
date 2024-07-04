import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import pandas as pd
from data.face_detection.ibug.face_detection import RetinaFacePredictor
from data.face_detection.ibug.face_detection.utils import SimpleFaceTracker
from data.datasets import convert_mp4_to_mp3, img_processing, pad_wav
from models.architectures import ResNet50, AudioModel, AVTmodel
from PIL import Image
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import time

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


class EmotionRecognition:
    def __init__(self):
        self.device = "cuda:0"
        self.load_models()

    def predict_emotion(self, path):
        self.dict_time = {}

        v_fss = self.load_video_frames(path)
        wav, a_fss = self.load_audio_features(path)
        t_fss = self.load_text_features(wav)
        pred_sc = self.predict_single_corpus(a_fss, v_fss, t_fss)
        pred_mc = self.predict_multi_corpus(a_fss, v_fss, t_fss)

        top_emotions_sc = self.get_top_emotions(pred_sc)
        top_emotions_mc = self.get_top_emotions(pred_mc)

        return top_emotions_sc, top_emotions_mc

    def get_top_emotions(self, predictions):
        EMOTIONS = [
            "Neutral",
            "Happy",
            "Sad",
            "Surprise",
            "Fear",
            "Disgust",
            "Angry",
        ]
        top_indices = np.argsort(predictions)[0][-2:][::-1]
        top_emotions = [
            f"{EMOTIONS[index]}: {predictions[0][index]:.2f}" for index in top_indices
        ]
        return ", ".join(top_emotions)

    def load_models(self):
        self.load_video_model()
        self.load_audio_model()
        self.load_text_model()
        self.load_avt_models()
        self.load_data_processor()

    def load_video_model(self):
        self.video_model = ResNet50(num_classes=7, channels=3)
        self.video_model.load_state_dict(
            torch.load("src/weights/FER_static_ResNet50_AffectNet.pt")
        )
        self.video_model.to(self.device).eval()
        self.face_detector = RetinaFacePredictor(
            threshold=0.8,
            device=self.device,
            model=RetinaFacePredictor.get_model("resnet50"),
        )
        self.face_tracker = SimpleFaceTracker(iou_threshold=0.4, minimum_face_size=0.0)

    def load_audio_model(self):
        path_audio_model = "DunnBC22/wav2vec2-base-Speech_Emotion_Recognition"
        self.processor = AutoFeatureExtractor.from_pretrained(path_audio_model)
        config = AutoConfig.from_pretrained(path_audio_model)
        num_classes = 7
        config.num_labels = num_classes
        self.audio_model = AudioModel.from_pretrained(
            path_audio_model, config=config, ignore_mismatched_sizes=True
        )
        self.audio_model.classifier = nn.Linear(
            config.classifier_proj_size, config.num_labels
        )
        self.audio_model.load_state_dict(torch.load("src/weights/MELD/audio.pth"))
        self.audio_model.to(self.device).eval()

    def load_text_model(self):
        path_text_model = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(path_text_model)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            path_text_model
        )
        self.text_model.load_state_dict(torch.load("src/weights/MELD/text.pth"))
        self.text_model.cuda().eval()
        self.s2t = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            chunk_length_s=30,
            device=self.device,
        )
        self.features = {}
        self.text_model.classifier.dense.register_forward_hook(
            self.get_activations("features")
        )

    def load_avt_models(self):
        num_classes = 7
        path_avt_sc_model = "src/weights/MELD/fusion.pth"
        path_avt_mc_model = "src/weights/LOCO/MELD_encoders/test_AFEW.pth"

        self.avt_sc_model = AVTmodel(
            512, 1024, 768, gated_dim=32, n_classes=num_classes, drop=0
        )
        self.avt_sc_model.load_state_dict(torch.load(path_avt_sc_model))
        self.avt_sc_model.cuda().eval()

        self.avt_mc_model = AVTmodel(
            512, 1024, 768, gated_dim=64, n_classes=num_classes, drop=0
        )
        self.avt_mc_model.load_state_dict(torch.load(path_avt_mc_model))
        self.avt_mc_model.cuda().eval()

    def load_data_processor(self):
        self.step = 2  # sec
        self.window = 4  # sec
        self.need_frames = 5
        self.sr = 16000

    def get_activations(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()

        return hook

    def load_video_frames(self, path):
        start_time = time.time()
        window_v = self.window * self.need_frames
        step_v = self.step * self.need_frames
        video_stream = cv2.VideoCapture(path)
        w = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        sec = frame_count / fps

        step = int(fps * 5 / 25)
        count_frame = 0
        dict_id_faces_images = {}

        while True:
            ret, fr = video_stream.read()
            if not ret:
                break
            if count_frame % step == 0:
                faces = self.face_detector(fr, rgb=False)
                tids = self.face_tracker(faces)
                for face, tid in zip(faces, tids):
                    startX, startY, endX, endY = face[:4].astype(int)
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w - 1, endX), min(h - 1, endY)
                    cur_face = fr[startY:endY, startX:endX]
                    cur_face = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    cur_face_copy = img_processing(Image.fromarray(cur_face))
                    id_face = str(tid - 1).zfill(2)
                    if id_face not in dict_id_faces_images:
                        dict_id_faces_images[id_face] = [cur_face_copy]
                    else:
                        dict_id_faces_images[id_face].append(cur_face_copy)

            count_frame += 1

        video_stream.release()
        self.face_tracker.reset()

        v_fss = torch.cat(dict_id_faces_images["00"], dim=0)

        with torch.no_grad():
            v_fss = self.video_model.extract_features(v_fss.to(self.device))
            v_fss = v_fss.cpu().numpy()

        segments_v = []
        for start_v in range(0, v_fss.shape[0] + 1, step_v):
            end_v = min(start_v + window_v, v_fss.shape[0])
            segment = v_fss[start_v:end_v, :]
            if end_v - start_v < step_v and start_v != 0:
                break
            segments_v.append(np.mean(segment, axis=0))

        segments_v = np.array(segments_v)

        v_fss = np.hstack((np.mean(segments_v, axis=0), np.std(segments_v, axis=0)))
        v_fss = torch.from_numpy(v_fss)
        v_fss = torch.unsqueeze(v_fss, 0)

        time_video = time.time() - start_time
        self.dict_time["time_video"] = sec % 60
        self.dict_time["time_feature_video"] = time_video % 60

        return v_fss

    def load_audio_features(self, path):
        start_time = time.time()
        window_a = self.window * self.sr
        step_a = self.step * self.sr

        wav = convert_mp4_to_mp3(path, self.sr)

        segments_a = []

        for start_a in range(0, len(wav) + 1, step_a):
            end_a = min(start_a + window_a, len(wav))
            if end_a - start_a < step_a and start_a != 0:
                break
            a_fss_chunk = wav[start_a:end_a]
            a_fss = pad_wav(a_fss_chunk, window_a)
            a_fss = torch.unsqueeze(a_fss, 0)
            a_fss = self.processor(a_fss, sampling_rate=self.sr)
            a_fss = a_fss["input_values"][0]
            segments_a.append(torch.from_numpy(a_fss))

        a_fss = torch.cat(segments_a)
        with torch.no_grad():
            a_fss = self.audio_model.extract_features(a_fss.to(self.device))
            a_fss = a_fss.cpu().numpy()

        a_fss = np.mean(a_fss, axis=1)
        a_fss = np.hstack((np.mean(a_fss, axis=0), np.std(a_fss, axis=0)))
        a_fss = torch.from_numpy(a_fss)
        a_fss = torch.unsqueeze(a_fss, 0)

        time_audio = time.time() - start_time
        self.dict_time["time_feature_audio"] = time_audio % 60
        return wav, a_fss

    def load_text_features(self, wav):
        start_time = time.time()
        text = self.s2t(wav.numpy(), batch_size=8)["text"]

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            _ = self.text_model(**inputs.to(self.device))
            t_fss = self.features["features"]
            self.features.clear()
        time_text = time.time() - start_time
        self.dict_time["time_feature_text"] = time_text % 60
        return t_fss

    def predict_single_corpus(self, a_fss, v_fss, t_fss):
        start_time = time.time()
        with torch.no_grad():
            pred_sc = self.avt_sc_model(
                a_fss.to(self.device), v_fss.to(self.device), t_fss.to(self.device)
            )
        pred_sc = nn.functional.softmax(pred_sc, dim=1).cpu().detach().numpy()
        time_sc = time.time() - start_time
        self.dict_time["time_pred_sc"] = time_sc % 60
        return pred_sc

    def predict_multi_corpus(self, a_fss, v_fss, t_fss):
        start_time = time.time()
        with torch.no_grad():
            pred_mc = self.avt_mc_model(
                a_fss.to(self.device), v_fss.to(self.device), t_fss.to(self.device)
            )
        pred_mc = nn.functional.softmax(pred_mc, dim=1).cpu().detach().numpy()
        time_mc = time.time() - start_time
        self.dict_time["time_pred_mc"] = time_mc % 60
        return pred_mc


# Example usage:
if __name__ == "__main__":
    folder = "src/test_video/"
    name_videos = [i for i in os.listdir(os.path.join(folder)) if i.endswith(".avi")]
    emotion_recognition = EmotionRecognition()
    path_true_data = "src/test_video/df_text_AFEW.csv"
    df_AFEW = pd.read_csv(path_true_data)

    for idx, path in enumerate(name_videos):
        print(f"{idx+1} / {len(name_videos)}")
        true_emo = df_AFEW[df_AFEW.name_file == path].emo.tolist()[0]
        pred_emotion_sc, pred_emotion_mc = emotion_recognition.predict_emotion(
            os.path.join(folder, path)
        )
        print("Name video: ", path)
        print(
            "Video duration sec: {:.2f}".format(
                emotion_recognition.dict_time["time_video"]
            )
        )
        print(
            "Recognition time sec: {:.2f}".format(
                sum(list(emotion_recognition.dict_time.values())[1:])
            )
        )
        print("True emotion:", true_emo)
        print("Two max predicted emotions of the Single-Corpus model:", pred_emotion_sc)
        print("Two max predicted emotions of the Multi-Corpus model:", pred_emotion_mc)
        print()
