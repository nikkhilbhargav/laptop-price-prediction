from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("catboost_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    brand = request.form["brand"]
    spec_rating = float(request.form["spec_rating"])
    processor = request.form["processor"]
    CPU = request.form["CPU"]
    Ram = request.form["Ram"]
    Ram_type = request.form["Ram_type"]
    ROM = request.form["ROM"]
    ROM_type = request.form["ROM_type"]
    GPU = request.form["GPU"]
    display_size = float(request.form["display_size"])
    resolution_width = float(request.form["resolution_width"])
    resolution_height = float(request.form["resolution_height"])
    OS = request.form["OS"]
    warranty = int(request.form["warranty"])

    data = pd.DataFrame({
        "brand":[brand],
        "spec_rating":[spec_rating],
        "processor":[processor],
        "CPU":[CPU],
        "Ram":[Ram],
        "Ram_type":[Ram_type],
        "ROM":[ROM],
        "ROM_type":[ROM_type],
        "GPU":[GPU],
        "display_size":[display_size],
        "resolution_width":[resolution_width],
        "resolution_height":[resolution_height],
        "OS":[OS],
        "warranty":[warranty]
    })

    prediction = model.predict(data)[0]

    return render_template(
        "result.html",
        prediction=round(prediction,2)
    )


if __name__ == "__main__":
    app.run(debug=True)