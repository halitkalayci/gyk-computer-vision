# Transfer Learning.


# ....

from ultralytics import YOLO


def train():
    model = YOLO("yolov8n.pt")
    model.train(data="data.yaml", epochs=1, imgsz=640, batch=8)

# Pythondaki main method.
# python main.py => Çalış
# başka bir dosyadan import edildiğinde çalışmaz.
if __name__ == "__main__":
    train()

