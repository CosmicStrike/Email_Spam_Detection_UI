from flask import Flask, request, render_template, session, redirect
import numpy as np
import pandas as pd


app = Flask(__name__)

df = pd.read_csv("cleaned.csv").head(100)


@app.route('/', methods=("POST", "GET"))
def html_table():
    return render_template('dataset.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
