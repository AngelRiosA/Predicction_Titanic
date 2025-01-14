from sqlalchemy import create_engine
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import datetime
import json
import io
import os
import base64
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import google.generativeai as genai
# from utils import get

app = Flask(__name__)

load_dotenv()

churro = os.environ.get("CHURRO")
engine = create_engine(churro)


def get_ts():
    
    timestamp = datetime.datetime.now().isoformat()
    return timestamp[0:19]

@app.route('/', methods=["GET"])
def formulario():
    return render_template('formulario.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Recogemos Los INPUTS
    pclass = int(request.form.get("pclass"))
    sex = int(request.form.get("sex"))
    age = int(request.form.get("age"))

    inputs = [pclass, sex, age]

    # Cargamos el modelo
    with open("titanic_model.pkl", "rb") as f:
        modelito = pickle.load(f)
    
    # Hacemos predicciones y montamos el TIMESTAMP
    outputs = modelito.predict([inputs])[0]
    timestamp = get_ts()

    # Lo subimos para arriba
    logs_to_parriba = pd.DataFrame({"inputs": [inputs], 
                                    "predictions": [outputs], 
                                    "timestamps": [timestamp]})
    logs_to_parriba.to_sql("predictions", con=engine, index=False, if_exists="append")


    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    
    fig = plt.figure()
    logs_leidos.predictions.value_counts().plot(kind="bar")
    plt.title(f"PREDICTIONS UP TO : {logs_leidos.timestamps.max()}")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    print("Outputs:", outputs)
    print("Logs leídos:", logs_leidos)

    return render_template('resultado.html', prediccion=outputs, grafica=img_base64)

#f"Hello Gemini, long time no see! I’ve created a prediction API for survival/non-survival "
#        f"based on the Titanic dataset. Here are the inputs:\n"
#        f" - PClass: {inputs[0]}\n"
#        f" - Sex: {inputs[1]} (0 = male, 1 = female)\n"
#        f" - Age: {inputs[2]}\n"
#        f" - Prediction: {prediction}\n\n"
#        f"Can you generate a brief, narrative, and adventurous description based on these inputs? "
#        f"Keep it concise (100-500 words) and include line breaks to improve readability."

@app.route("/results", methods=["GET"])
def results():
    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    return json.loads(logs_leidos.to_json(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)