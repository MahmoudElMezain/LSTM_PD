import torch
import pandas as pd
import numpy as np
import cv2
import time

print(torch.version.cuda)
print(torch.cuda.is_available())

# Load YOLOv5 model
model = torch.hub.load('YOLOv5_Model', 'custom', path="MARKER.pt", source='local')

# Load video
cap = cv2.VideoCapture("Test1.mp4")
final_table = pd.DataFrame(columns=['time', 'xcenter', 'ycenter', 'FLAG'])

timer = 0
x_center_old = 10000
y_center_old = 10000
flag = ""

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(w, h)

# Setup video writer
save = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = RGB[::-1, :, :]  # Flip vertically for YOLO format

    prediction = model(img)
    results = prediction.pandas().xyxy[0]

    if results.empty:
        new_row = pd.DataFrame([{'time': timer, 'xcenter': None, 'ycenter': None, 'FLAG': 'ERROR'}])
    else:
        upper_prediction_table = results.loc[results['ymax'] > 600]
        result = upper_prediction_table.loc[upper_prediction_table['xmax'] == upper_prediction_table['xmax'].min()]

        x1 = int(result['xmin'])
        y1 = h - int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = h - int(result['ymax'])

        x_center = float((result['xmin'] + result['xmax']) / 2)
        y_center = float((result['ymin'] + result['ymax']) / 2)

        if abs(y_center - y_center_old) > 25 or abs(x_center - x_center_old) > 25:
            flag = "Flagged"
        else:
            flag = "   "

        x_center_old = x_center
        y_center_old = y_center

        new_row = pd.DataFrame([{
            'time': timer,
            'xcenter': x_center,
            'ycenter': y_center,
            'FLAG': flag
        }])

        # Draw bounding box and center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 8)
        cv2.circle(frame, (int(x_center), h - int(y_center)), radius=4, color=(0, 0, 255), thickness=-1)

    # Append row and update time
    final_table = pd.concat([final_table, new_row], ignore_index=True)
    timer += 0.1

    # Display and save frame
    cv2.imshow("Capture", cv2.resize(frame, (960, 520)))
    save.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
save.release()
cv2.destroyAllWindows()

# Save CSV
final_table.to_csv('Markers.csv', index=False)
