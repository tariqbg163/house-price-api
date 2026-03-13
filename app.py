from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load ML model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # get value from form
    size = float(request.form["size"])

    # model prediction
    prediction = model.predict([[size]])

    result = round(prediction[0],2)

    return render_template("index.html", prediction_text=f"Predicted House Price: ${result}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)