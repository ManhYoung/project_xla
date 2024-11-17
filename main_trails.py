import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("H:\AI\project_xla\Resources\Videos\Video7.mp4")

track_history = defaultdict(lambda : [])

while True:
    ret,frame = cap.read()
    if ret:
        results = model.track(source=frame, persist= True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color= (230,0,0), thickness=10)

        cv2.imshow("YOLO11 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()