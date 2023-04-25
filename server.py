from flask import Flask, request, render_template, session, redirect
import numpy as np
import pandas as pd
import os
import flask
import dataCleaning
import training


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if os.path.exists(os.path.join(flask.current_app.root_path, "datasets", 'data.csv')):
        df = pd.read_csv(os.path.join(
            flask.current_app.root_path, "datasets", 'data.csv')).head(100)
        length = df.size
        return render_template('index.html',  tables=[df.to_html(classes='data', index=False)], titles=df.columns.values, length=length)
    else:
        return render_template("index.html")


@app.route('/clean', methods=("POST", "GET"))
def html_table():
    clean = request.args.get('clean')
    if clean and os.path.exists(os.path.join(flask.current_app.root_path, "datasets", 'data.csv')):
        df = pd.read_csv(os.path.join(
            flask.current_app.root_path, "datasets", 'data.csv'))
        dataCleaning.initialize()
        df = dataCleaning.clean(df)
        return render_template('dataset.html',  tables=[df.to_html(classes='data', index=False)], titles=df.columns.values)

    elif os.path.exists(os.path.join(flask.current_app.root_path, "datasets", 'cleaned.csv')):
        df = pd.read_csv(os.path.join(
            flask.current_app.root_path, "datasets", 'cleaned.csv'))
        return render_template('dataset.html',  tables=[df.to_html(classes='data', index=False)], titles=df.columns.values)
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
        if x:
            return {"status": "success", "result": "spam"}
        else:
            return {"status": "success", "result": "not spam"}


@app.route("/model")
def model():
    if os.path.exists("model.pickle"):
        return render_template("model.html")
    return render_template("model.html", redirect=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
