import torch
import pandas as pd
import numpy as np
import cv2
import time

print(torch.version.cuda)
print(torch.cuda.is_available())

model = torch.hub.load('./yolov5', 'custom', path="./yolov5/runs/train/exp/weights/FISH.pt",source='local')  # Import YOLO model, by providing path to folder and path to weights
cap = cv2.VideoCapture("Test1.mp4")  # Inserting Test Video
final_table = pd.DataFrame(columns=['time','xcenter','ycenter'])   #Creates final table with xcenter and ycenter values
timer = 0

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #returns captured frame width
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #returns captured frame height
print(w,h)
fps = int(cap.get(cv2.CAP_PROP_FPS))    #returns fps
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #returns number of frames

while True:
 ret,frames = cap.read()  # Current Frame of the input video
 if (ret == 0):
     break

 frame = cv2.resize(frames, (1920,1080))
 RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #Opencv reads the image in BGR not RGB, so it has to be converted
 img = RGB[::-1, :, :]  #flips the image so the origin is at the bottom left instead of top left

 prediction = model(img) #inference
 results = prediction.pandas().xyxy[0]    #dataframe containing bounding box coordinates and the confidence score
 if (results.empty):  # If no detections, add row to the dataframe indication no detections
        final_table= final_table.append({'time':timer, 'xcenter':'NONE', 'ycenter':'NONE'}, ignore_index=True)
        timer = timer + 0.1
 else:
        result = results.loc[results['confidence'] == results['confidence'].max()]
        x1 = int(result['xmin'])
        y1 = h-int(result['ymin'])  #flip the y-axis back for the display
        x2 = int(result['xmax'])
        y2 = h-int(result['ymax'])
        x_center = float((result['xmin'] + result['xmax']) / 2)  # obtains the x-coordinates of the midpoint of the bounding box
        y_center = float((result['ymin'] + result['ymax']) / 2)  # obtains the y-coordinates of the midpoint of the bounding box
        final_table = final_table.append({'time': timer, 'xcenter': x_center, 'ycenter': y_center},ignore_index=True)  # adds new row with the selected values
        timer = timer + 0.1
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draws rectangle around object
        #cv2.circle(frame, (int(x_center), h-int(y_center)), radius=10, color=(0, 0, 255), thickness=-1) #Draw dot at the center of the robot
        cv2.imshow("Capture", cv2.resize(frame, (1080, 720)))

 if cv2.waitKey(1) & 0xFF == ord("q"):  # Breaks loop if q is pressed
   break

cap.release()
cv2.destroyAllWindows()

print(final_table)
final_table.to_csv('Output_Sequence.csv', index=False) #Extract the pandas dataframe as a csv file