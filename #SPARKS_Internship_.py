#SPARKS_Internship_

import cv2 
import numpy 
import matplotlib.pyplot as plt

a= "C:\\Users\\lenovo\\Desktop\\Sparks\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
b=  "C:\\Users\\lenovo\\Desktop\\Sparks\\frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(b,a)
classlabels=[]
c= "C:\\Users\\lenovo\\Desktop\\Sparks\\Labels.txt"
with open(c,'rt') as fpt:
    classlabels= fpt.read().rstrip('\n').split('\n')

img = cv2.imread("C:\\Users\\lenovo\\Downloads\\4f49445843972d84-600x338.jpg")
plt.imshow(img)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#Code below this comment is for object detection in an image in real-time.




Classindex,confidence,bbox= model.detect(img,confThreshold=0.585)#accuracyindex
font_scale=3
font= cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(Classindex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classlabels[ClassInd-1],(boxes[0]+10, boxes[1]+40), font, fontScale= font_scale,color=(0,0,255),thickness=3)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()



#Code below this comment is for object detection in a video in real time.



cap = cv2.VideoCapture("C:\\Users\\lenovo\\Downloads\\pexels-cottonbro-8629573.mp4")
if not cap.isOpened():
    cap = cv2.VideoCapture()
if not cap.isOpened():
    raise IOError('Cannot open video')
font_scale=1.25
while(True):
    ret,frame= cap.read()
    frame= cv2.resize(frame,(700,700))
    Classindex,confidence,bbox= model.detect(frame,confThreshold=0.501)
    if len(Classindex)!=0:
        for ClassInd,conf,boxes in zip(Classindex.flatten(),confidence.flatten(),bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classlabels[ClassInd-1],(boxes[0]+10, boxes[1]+40), font, fontScale= font_scale,color=(0,0,255),thickness=2)

    cv2.imshow('object detection tutorial',frame)
    

    if cv2.waitKey(2) and ord('q')==0xFF:
        break
cap.release()
cv2.destroyAllWindows()






