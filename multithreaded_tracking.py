import threading
import cv2
from ultralytics import YOLO

MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]

SOURCES = ["H:\AI\project_xla\Resources\Videos\Video7.mp4","H:\AI\project_xla\Resources\Videos\Video8.mp4"]

def run_tracker_in_thread(model_name, file_name):
    model = YOLO(model_name)
    results = model.track(source= file_name, save = True, stream= True, show= True)
    for r in results:
        pass

tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon= True)
    tracker_threads.append(thread)
    thread.start()

for thread in tracker_threads:
    thread.join()

cv2.destroyAllWindows