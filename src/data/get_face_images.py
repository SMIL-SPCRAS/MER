import os
import cv2

from tqdm import tqdm

from face_detection.ibug.face_detection import RetinaFacePredictor
from face_detection.ibug.face_detection.utils import SimpleFaceTracker


class VideoPredictor:
    def __init__(self):
        super().__init__()
        self.video_stream = None
        self.device = "cuda:0"
        self.model = None
        self.count_frame = None
        self.init_predictor()

    def init_path(self, path):
        self.video_stream = cv2.VideoCapture(path)
        self.w = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def init_predictor(self):
        self.model = RetinaFacePredictor(
            threshold=0.8,
            device=self.device,
            model=RetinaFacePredictor.get_model("resnet50"),
        )
        self.face_tracker = SimpleFaceTracker(iou_threshold=0.4, minimum_face_size=0.0)

    def __del__(self):
        if self.video_stream is not None:
            self.video_stream.release()

    def process(self, path, save_path):
        self.count_frame = 0
        self.init_path(path)
        # print(path)
        name_file = os.path.basename(path)

        while True:
            ret, fr = self.video_stream.read()
            if not ret:
                break

            n_img = str(self.count_frame).zfill(6)
            pred = self.model(fr, rgb=False)
            tids = self.face_tracker(pred)

            for pred, tid in zip(pred, tids):
                startX, startY, endX, endY = pred[:4].astype(int)
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(self.w - 1, endX), min(self.h - 1, endY)
                cur_fr = fr[startY:endY, startX:endX]
                c_path = os.path.join(save_path, name_file[:-4], str(tid - 1).zfill(2))
                os.makedirs(c_path, exist_ok=True)
                cv2.imwrite(os.path.join(c_path, n_img + ".jpg"), cur_fr)
            self.count_frame += 1

        self.face_tracker.reset()


if __name__ == "__main__":

    folder_videos = "D:/Databases/AFEW/"
    folder_save_images = "C:/Work/Faces/IS_2/AFEW_faces/"

    detect = VideoPredictor()

    """
    If there are more than two unique faces after the detection
    of the face images, you should determine a target face.
    """

    for subset in ["Train_AFEW", "Val_AFEW"]:
        for emo in tqdm(
            ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        ):
            name_videos = [
                i
                for i in os.listdir(os.path.join(folder_videos, subset, emo))
                if i.endswith(".avi")
            ]
            path_save_images = os.path.join(folder_save_images, subset, emo)
            for name_video in name_videos:
                # print(name_video)
                curr_video = os.path.join(folder_videos, subset, emo, name_video)
                detect.process(curr_video, path_save_images)
