import torch
import pandas as pd
import cv2

# Print CUDA info
print(torch.version.cuda)
print(torch.cuda.is_available())

# Load YOLOv5 model
model = torch.hub.load('YOLOv5_Model', 'custom', path="FISH.pt", source='local')

# Load video
cap = cv2.VideoCapture("Test1.mp4")

# Create empty tracking table
final_table = pd.DataFrame(columns=['time', 'xcenter', 'ycenter'])
timer = 0

# Video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Original video resolution: {w}x{h}, FPS: {fps}, Total frames: {n_frames}")

# Resize target
target_width = 1920
target_height = 1080

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (target_width, target_height))

    # Convert BGR to RGB and flip vertically
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = RGB[::-1, :, :]  # Flip vertically (bottom-left origin)

    # YOLO Inference
    prediction = model(img)
    results = prediction.pandas().xyxy[0]

    if results.empty:
        new_row = pd.DataFrame([{'time': timer, 'xcenter': None, 'ycenter': None}])
    else:
        # Pick detection with highest confidence
        result = results.loc[results['confidence'] == results['confidence'].max()]

        # Get bounding box coordinates
        x1 = int(result['xmin'])
        y1 = target_height - int(result['ymin'])  # Flip back
        x2 = int(result['xmax'])
        y2 = target_height - int(result['ymax'])  # Flip back

        # Compute center
        x_center = float((result['xmin'] + result['xmax']) / 2)
        y_center = float((result['ymin'] + result['ymax']) / 2)
        y_center_display = target_height - y_center  # Flip back for display

        new_row = pd.DataFrame([{
            'time': timer,
            'xcenter': x_center,
            'ycenter': y_center
        }])

        # Draw box and center point
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (int(x_center), int(y_center_display)), radius=10, color=(0, 0, 255), thickness=-1)

    # Add to table and advance timer
    final_table = pd.concat([final_table, new_row], ignore_index=True)
    timer += 0.1

    # Display
    cv2.imshow("Capture", cv2.resize(frame, (1080, 720)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
final_table.to_csv('Output_Sequence.csv', index=False)
print(final_table)
