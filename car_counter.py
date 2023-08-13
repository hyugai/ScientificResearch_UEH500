# Supporting Lib
import os
import cv2
import math
import random
import cvzone
#from scipy.optimize import linear_sum_assignment as linear_assignment

# Class_list
cls_file=open('coco-classes.txt','r')
cls_data=cls_file.read()
class_name=cls_data.split('\n')
cls_file.close()

# Colors for bbox
colors=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for j in range(10)]

# Detection Lib
from ultralytics import YOLO

# Tracking Lib
from tracker import Tracker

# Create pre-trained instance of YOLO and one for tracker
model=YOLO('yolov8m.pt')
tracker=Tracker()

# Video input
video_path=os.path.join('.','data','v7.mp4')
video_cap=cv2.VideoCapture(video_path)

ret,frame=video_cap.read()

# Draw the line
limits=[150,750,1920,750]

# Count
total_count=[]

while ret:

    # Feed the model to get data as list-elements in a list
    frame_detections=model(frame)

    for frame_detection in frame_detections:

        # DETECTION #######
        detections=[]
        # In this for loop we're working with each data one by one until the end
        for data in frame_detection.boxes.data.tolist():
            x_min,y_min,x_max,y_max,conf_score,cls_id=data

            id=int(cls_id)
            conf=math.ceil((conf_score)*100)/100
            class_current=class_name[id]
            if class_current=='car' or class_current=='truck' or class_current=='motorbike' or class_current=='bus' and conf>0.5:
                detections.append([int(x_min),int(y_min),int(x_max),int(y_max),conf])
        # Output of detections
        #print(detections)

        # TRACKING #######
        # Create bbox for each object in frame and assign them a specific ID
        # In next loop, the program will compare information input in each iteration to track. It still uses some old information by the way
        tracker.update(frame,detections)

        # Draw the line
        line = cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        # Get ID and the data for bbox by using ATTRIBUTES .bbox and .track_id
        # .tracks is used to access the list 'tracks' containing specific data called 'Track'
        # Variable 'track' contains that one 
        for track in tracker.tracks:
            bbox=track.bbox
            # x1=xmin, y1=ymin, x2=xmax, y2=ymax
            x1,y1,x2,y2=bbox
            track_id=track.track_id

            # Draw bbox and ID for each object
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(colors[track_id % len(colors)]),3)
            cvzone.putTextRect(frame,f'{track_id}',(max(0, int(x1)), max(35, int(y1)-5)), scale=1.5, thickness=2,offset=3)

            # Draw the circle in middle of the bbox
            cx=(x1+x2)/2
            cy=(y1+y2)/2
            cv2.circle(frame,(int(cx), int(cy)), 5, (colors[track_id % len(colors)]), cv2.FILLED)

            # COUNT #######
            if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[1]+20:
                if total_count.count(track_id)==0:
                    total_count.append(track_id)

                    # Change the color of the line when a car crossing
                    line = cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 255), 5)

        # PRINT THE COUNT ON THE SCREEN
        cv2.putText(frame, 'Count:' +str(len(total_count)), (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)

    # Renew the frame
    ret,frame=video_cap.read()

video_cap.release()
cv2.destroyAllWindows()
