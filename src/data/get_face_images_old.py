import os
import cv2
import numpy as np

from tqdm import tqdm
from scipy.spatial import distance as dist
from collections import OrderedDict

from face_detection.ibug.face_detection import RetinaFacePredictor


class CentroidTracker:
    """
    https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """

    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    @staticmethod
    def get_area(rect):
        return (rect[2] - rect[0]) * (rect[3] - rect[1])

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        rects = sorted(rects, key=self.get_area, reverse=True)

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        rows = None
        cols = None

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return rects, rows, cols


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
            rects = []
            pred = self.model(fr, rgb=False)
            if len(pred) > 0:
                for curr_bbox in pred[:, :4].astype("int"):
                    startX, startY, endX, endY = curr_bbox
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(self.w - 1, endX), min(self.h - 1, endY)
                    rects.append([startX, startY, endX, endY])
            if rects:
                rects, rows, cols = ct.update(rects)
                cols_was = []
                if rows is None:
                    for id_face in range(len(rects)):
                        cur_fr = fr[
                            rects[id_face][1] : rects[id_face][3],
                            rects[id_face][0] : rects[id_face][2],
                        ]
                        c_path = os.path.join(
                            save_path, name_file[:-4], str(id_face).zfill(2)
                        )
                        os.makedirs(c_path, exist_ok=True)
                        cv2.imwrite(os.path.join(c_path, n_img + ".jpg"), cur_fr)

                else:
                    for row, col in zip(rows, cols):
                        if col not in cols_was:
                            cols_was.append(col)
                            cur_fr = fr[
                                rects[col][1] : rects[col][3],
                                rects[col][0] : rects[col][2],
                            ]
                            c_path = os.path.join(
                                save_path, name_file[:-4], str(row).zfill(2)
                            )
                            os.makedirs(c_path, exist_ok=True)
                            cv2.imwrite(os.path.join(c_path, n_img + ".jpg"), cur_fr)

            self.count_frame += 1


if __name__ == "__main__":

    folder_videos = "D:/Databases/AFEW/"
    folder_save_images = "C:/Work/Faces/IS/AFEW_faces/"

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
                ct = CentroidTracker()
                detect.process(curr_video, path_save_images)
