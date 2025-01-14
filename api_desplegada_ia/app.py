from sqlalchemy import create_engine
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import datetime
import json
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)


churro = "postgresql://postgres:postgres@34.140.73.76:5432/postgres"
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


    return render_template('resultado.html', prediccion=outputs, grafica=img_base64)

@app.route("/results", methods=["GET"])
def results():
    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    return json.loads(logs_leidos.to_json(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)