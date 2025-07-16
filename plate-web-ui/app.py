from flask import Flask, request, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid

app = Flask(__name__)

# Static klasörünü oluştur
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# TensorFlow Lite modelini yükle
interpreter = tf.lite.Interpreter(model_path="models/plate_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def draw_bounding_boxes(image, detections, threshold=0.5):
    """
    Resim üzerine bounding box'ları çizer
    """
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        # Model çıkışının formatını kontrol et
        print(f"Detection shape: {detections.shape}")
        print(f"Detection type: {type(detections)}")
        
        # Eğer detections boş veya scalar ise, orijinal resmi döndür
        if detections.size == 0 or len(detections.shape) == 0:
            print("No detections found or scalar output")
            return draw_image
            
        # Farklı shape formatlarını handle et
        if len(detections.shape) == 3:
            detections = detections[0]  # Batch boyutunu kaldır
        elif len(detections.shape) == 1:
            # Tek boyutlu array ise reshape et veya işleme
            if detections.size < 5:
                print("Insufficient detection data")
                return draw_image
            detections = detections.reshape(1, -1)
        
        # Eğer 2D değilse, orijinal resmi döndür
        if len(detections.shape) != 2:
            print("Unexpected detection format")
            return draw_image
            
        height, width = image.size[1], image.size[0]
        detection_count = 0
        
        for detection in detections:
            if len(detection) >= 5:
                confidence = detection[4]
                if confidence > threshold:
                    detection_count += 1
                    # YOLO formatından pixel koordinatlarına çevir
                    x_center, y_center, box_width, box_height = detection[:4]
                    
                    # Normalize edilmiş koordinatları pixel koordinatlarına çevir
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height
                    
                    # Köşe koordinatlarını hesapla
                    x1 = int(x_center - box_width / 2)
                    y1 = int(y_center - box_height / 2)
                    x2 = int(x_center + box_width / 2)
                    y2 = int(y_center + box_height / 2)
                    
                    # Bounding box çiz
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    
                    # Güven skorunu yazdır
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    
                    text = f"{confidence:.2f}"
                    draw.text((x1, y1-25), text, fill='red', font=font)
        
        print(f"Drew {detection_count} bounding boxes")
        
    except Exception as e:
        print(f"Error in draw_bounding_boxes: {str(e)}")
        # Hata durumunda orijinal resmi döndür
        pass
    
    return draw_image

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        
        # Dosya yüklü mü kontrol et
        if file.filename == '':
            return render_template("index.html", error="Lütfen bir dosya seçin!")
        
        # Benzersiz dosya adı oluştur
        unique_id = str(uuid.uuid4())
        original_filename = f"{unique_id}_original.jpg"
        result_filename = f"{unique_id}_result.jpg"
        
        # Resmi PIL ile aç ve işle
        original_image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Orijinal resmi kaydet
        original_image.save(f"static/uploads/{original_filename}")
        
        # Model için resmi yeniden boyutlandır
        model_image = original_image.resize((640, 640))
        img_array = np.array(model_image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Model ile tahmin yap
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Bounding box'ları çiz
        result_image = draw_bounding_boxes(original_image, output_data)
        result_image.save(f"static/results/{result_filename}")
        
        # Tespit sayısını güvenli bir şekilde hesapla
        detection_count = 0
        try:
            if len(output_data.shape) > 0 and output_data.size > 0:
                # Shape'e göre farklı işlemler
                if len(output_data.shape) == 3:
                    data = output_data[0]
                elif len(output_data.shape) == 2:
                    data = output_data
                else:
                    data = None
                
                if data is not None and len(data.shape) == 2:
                    for detection in data:
                        if len(detection) >= 5 and detection[4] > 0.5:
                            detection_count += 1
        except Exception as e:
            print(f"Error counting detections: {str(e)}")
            detection_count = 0
        
        # Sonuç sayfasına yönlendir
        return render_template("result.html", 
                             original_image=f"uploads/{original_filename}",
                             result_image=f"results/{result_filename}",
                             detections=detection_count)
        
    except Exception as e:
        return render_template("index.html", error=f"Hata oluştu: {str(e)}")

@app.route("/about")
def about():
    return render_template("about.html")

# Flask'ın çalıştırılması için gerekli olan kod
if __name__ == "__main__":
    app.run(debug=True)


# Bireysel ödev ->
# Bu websitesini yolov8 transfer learning ile eğittiğimiz model üzerinden kullanılacak biçime getirip UI/UX istediğiniz gibi düzenleyin.
# Websitesinde varsa gördüğünüz performans problemlerini gideriniz.