# from ultralytics import YOLO

# #load the YOLO11 Model
# model  = YOLO("yolo11n.pt")

# #results = model.track(source = "H:\AI\project_xla\Resources\Videos\Video7.mp4", show = True, save = True)

# #tracking with Byte-Track
# results = model.track(source= "H:\AI\project_xla\Resources\Videos\Video9.mp4", show = True, save = True, tracker = "bytetrack.yaml", conf = 0.20, iou = 0.3)

#----------------------------------------------#
#Python Script using OpenCV-Python (cv2) to run

import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolo11n.pt")
names = model.model.names

cap = cv2.VideoCapture("H:\AI\project_xla\Resources\Videos\Video5.mp4")
# line_pts = [(300, 1000), (2500, 1000)]
while True: 
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist= True)
        
        # speed_obj = solutions.SpeedEstimator(
        #     reg_pts=line_pts,
        #     names=names,
        #     view_img=True,
        # )
        # frame = speed_obj.estimate_speed(frame, results)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()