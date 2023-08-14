from ultralytics import YOLO
from sort import *
import cv2
import cvzone
import math


#import video
cap=cv2.VideoCapture('traffic.mp4')

#creating a fucntion
model=YOLO('..\YOLO_Weights\yolov8l.pt')
#model=YOLO('yolov5n.pt')

class_names=['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
            'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
            'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
            'handbag','tie','suitcase','frisbee','skis','snowboard','tennis racket','bottle','wine glass','cup',
            'fork','knife','spoon','bowl','banana','apple','sandwich','orange','brocoli',
            'carrot','hot dog','pizza','donut','cake','chair','sofa','pottendplant','bed',
            'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
            'microwave','oven','toaster','sink','regfrigerator','book','clock','vase','scissors',
             'teddy bear','hair drier','toothbrush']


#tracking, tracker is a variable that will reset after those parameter
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

#creating data for the line
limits1=[0,250,852,250]
limits2=[0,200,852,200]

total_count=[]

ret, frame = cap.read()

fourcc=cv2.VideoWriter_fourcc(*'MP4V')
width=cap.get(3)
height=cap.get(4)
video_out=cv2.VideoWriter(1983148141,24,(int(width),int(height)))

while ret:

    results=model(frame,stream=True)

    detections=np.empty((0,5))

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
                #data we need to draw a rectangle
                #we need to convert cusor data from YOLO (x1,y1,x2,y2) to integer data
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1

                #confidence
            conf=math.ceil((box.conf[0])*100)/100
                #class_names
            cls=int(box.cls[0])

            current_cls=class_names[cls]

            if current_cls=='car' or current_cls== 'bus' or current_cls=='truck' or current_cls=='motorbike' and conf>0.3:

                    #the output of those above code is array[x1,y1,x2,y2,cof]
                current_array=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,current_array))

    results_tracker=tracker.update(detections)
        #the first output the tracker will assign certain id for the objects

        #draw a line
    line1=cv2.line(frame,(limits1[0],limits1[1]),(limits1[2],limits1[3]),(0,0,255),5)
    line2=cv2.line(frame, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 5)

        #draw a center for each boxes
    cx,cy=x1+w//2,y1+h//2
    cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)

    for result in results_tracker:
            #get the id from the tracker assining and store it in results_tracker
        x1,y1,x2,y2,id=result
        print(result)

            #creating bounding box from here not above
        x1, y1, x2, y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=3, colorR=(255,0,0))
        cvzone.putTextRect(frame, f'{id}', (max(0, x1), max(35, y1 - 20)), scale=2, thickness=2,offset=3)

            # draw a center for each boxes
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            #counter
        if (limits1[0]<cx<limits1[2] and limits1[1]-20<cy<limits1[1]+20) :
            if total_count.count(id)==0:
                total_count.append(id)
                    #changing the color of the line when counting
                line1= cv2.line(frame, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 5)
            elif (limits2[0]<cx<limits2[2] and limits2[1]-20<cy<limits2[1]+20):
                if total_count.count(id) == 0:
                    total_count.append(id)
                line2 = cv2.line(frame, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 5)
            else:
                continue


        #showing counter
    cv2.putText(frame,str(len(total_count)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)

    video_out.write(frame)

    ret,frame=cap.read()

    print('Total cars in the main road: ',len(total_count))

cap.release()
video_out.release()
cv2.destroyAllWindows()