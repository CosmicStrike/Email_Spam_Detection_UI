from flask import Flask, request, render_template, session, redirect
import numpy as np
import pandas as pd
import flask
import os
import dataCleaning
app = Flask(__name__)

<<<<<<< HEAD
df = pd.read_csv("cleaned.csv").head(100)
=======

@app.route("/", methods=["GET", "POST"])
def index():
    if os.path.exists(os.path.join(flask.current_app.root_path, "datasets", 'data.csv')):
        df = pd.read_csv(os.path.join(
            flask.current_app.root_path, "datasets", 'data.csv')).head(100)
        length = df.size
        return render_template('index.html',  tables=[df.to_html(classes='data', index=False)], titles=df.columns.values, length=length)
    else:
        return render_template("index.html")
>>>>>>> 52bab62842611cc15d3919b50fd1433c315d62ef


@app.route('/clean', methods=("POST", "GET"))
def html_table():
<<<<<<< HEAD
    return render_template('dataset.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
=======
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


@app.route("/upload", methods=["POST"])
def upload_csv():
    file = request.files["file"]
    file.save(os.path.join(flask.current_app.root_path, "datasets", 'data.csv'))
    return {"status": "file successfully uploaded"}
>>>>>>> 52bab62842611cc15d3919b50fd1433c315d62ef


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
