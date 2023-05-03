from flask import Flask, request, render_template, session, redirect
import numpy as np
import pandas as pd
import os
import flask
import dataCleaning
import training
import seaborn as sns
from io import BytesIO
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if os.path.exists(os.path.join(flask.current_app.root_path, "datasets", 'data.csv')):
        df = pd.read_csv(os.path.join(
            flask.current_app.root_path, "datasets", 'data.csv'))
        length = df.size
        df = df.head(100)
        return render_template('index.html',  tables=[df.to_html(classes='data', index=False)], titles=df.columns.values, length=length)
    else:
        return render_template("index.html")


@app.route('/clean', methods=("POST", "GET"))
def html_table():
    clean = request.args.get('clean')

    if os.path.exists(os.path.join(flask.current_app.root_path, "datasets", 'data.csv')):
        df = pd.read_csv(os.path.join(
            flask.current_app.root_path, "datasets", 'data.csv'))
        if os.path.exists(os.path.join(flask.current_app.root_path, "datasets", 'cleaned.csv')):
            df2 = pd.read_csv(os.path.join(
                flask.current_app.root_path, "datasets", 'cleaned.csv'))
        else:
            dataCleaning.initialize()
            df2 = dataCleaning.clean(df)

        if clean:
            dataCleaning.initialize()
            df2 = dataCleaning.clean(df)

        plt.switch_backend("agg")
        plot = sns.histplot(data=df, x=df2.Length, hue="v1").get_figure()

        buf = BytesIO()
        plot.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return render_template('dataset.html',  tables=[df2.to_html(classes='data', index=False)], titles=df.columns.values, figure=data)
    else:
        return render_template('dataset.html')


@app.route("/upload", methods=["POST"])
def upload_csv():
    file = request.files["file"]
    file.save(os.path.join(flask.current_app.root_path, "datasets", 'data.csv'))
    return {"status": "success"}


@app.route("/train", methods=["GET"])
def train():
    df = pd.read_csv(os.path.join(
        flask.current_app.root_path, "datasets", 'cleaned.csv'))
    acc = training.model_fit_and_test(df)

    return {"status": "success", "accuracy": acc}


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = flask.request.json
    model = training.load_model()
    if model:
        x = data["email"]
        x = training.predict(x)
        return {"status": "success", "result": int(x[0][0]), "prob": list(x[1][0])}


@app.route("/model")
def model():
    if os.path.exists("model.pickle"):
        return render_template("model.html")
    return render_template("model.html", redirect=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
