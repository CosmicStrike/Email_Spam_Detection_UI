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
from sklearn.feature_extraction.text import TfidfVectorizer

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
        plot = sns.histplot(data=df2, x=df2.Length, hue="labels").get_figure()

        buf = BytesIO()
        plot.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")

        s, h = DrawPieChart(df2, 10)

        return render_template('dataset.html',  tables=[df2.to_html(classes='data', index=False)], titles=df.columns.values, figure=data, spamchart=s, hamchart=h)
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


def DrawPieChart(cleanedDf: pd.DataFrame, limit):
    hamData = cleanedDf.loc[cleanedDf['labels'] == 0]
    spamData = cleanedDf.loc[cleanedDf['labels'] == 1]

    hamTf = TfidfVectorizer()
    spamTf = TfidfVectorizer()

    hamTf.fit_transform(
        hamData['Processed_Text'].values.astype('U'))
    spamTf.fit_transform(
        spamData['Processed_Text'].values.astype('U'))

    hamWeights = (np.array(hamTf.idf_))
    hamWords = (np.array(hamTf.get_feature_names_out()))
    spamWeights = (np.array(spamTf.idf_))
    spamWords = (np.array(spamTf.get_feature_names_out()))

    colorPalette = sns.color_palette("bright")[0:limit]
    ham = pd.DataFrame(data={'weights': hamWeights, 'words': hamWords}).sort_values(
        'weights', ascending=False).drop_duplicates(['weights'])
    spam = pd.DataFrame(data={'weights': spamWeights, 'words': spamWords}).sort_values(
        'weights', ascending=False).drop_duplicates(['weights'])
    weightHam = np.array(ham['weights'])
    wordsHam = np.array(ham['words'])

    weightSpam = np.array(spam['weights'])
    wordsSpam = np.array(spam['words'])
    plt.clf()

    plt.pie(weightHam[1:limit],
            labels=wordsHam[1:limit],
            colors=colorPalette,
            autopct='%.0f%%')
    hampie = plt.gcf()
    buf = BytesIO()
    hampie.savefig(buf, format="png")
    hampie = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.clf()
    plt.pie(weightSpam[1:limit],
            labels=wordsSpam[1:limit],
            colors=colorPalette,
            autopct='%.0f%%')
    buf = BytesIO()
    spampie = plt.gcf()
    spampie.savefig(buf, format="png")
    spampie = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.clf()

    return spampie, hampie


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
