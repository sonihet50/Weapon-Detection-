import cv2
from ultralytics import YOLO


# model = YOLO("yolov8n.pt")
model = YOLO("yolo26n.pt")

# cap = cv2.VideoCapture("thieves.mp4")
cap = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)


while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("thieves",annotated_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
