from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")

image_path = "data/test/test.jpg"

image = cv2.imread(image_path)

result = model.predict(image)

print(result[0].boxes)

boxes = result[0].boxes

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = box.conf[0]
    
    if conf > 0.75:
        cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        label = f"Plaka - {conf:.2f}"
        cv2.putText(image, label, (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# xywh => x,y width,height