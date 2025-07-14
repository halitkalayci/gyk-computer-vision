from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=["GET"]) # Webden gelen url sonundaki kısım ne ile başlarsa bu fonksiyon çalışır?
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    print(file)
    return "Görsel alındı"


# Flask'ın çalıştırılması için gerekli olan kod
# Yalnızca python app.py ile çalıştırılabilir
if __name__ == "__main__":
    app.run(debug=True)