import os
from flask import Flask, render_template, request, send_file
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from flask_mysqldb import MySQL
import pickle
import numpy as np
import yaml
import datetime
import pandas as pd



app = Flask(__name__)


db = yaml.safe_load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']
mysql = MySQL(app)



def predict(values, dic):
    # diabetes
    if len(values) == 8:
        model1 = pickle.load(open('models/diabetes.pkl','rb'))
        model2 = pickle.load(open('models/rb.pkl','rb'))
        values = model2.transform([values])
        values = np.asarray(values)
        return model1.predict(values.reshape(1, -1))[0]

    # breast_cancer
    elif len(values) == 22:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        model2 = pickle.load(open('models/scaler.pkl','rb'))
        values = model2.transform([values])
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # heart disease
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # kidney disease
    elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # liver disease
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/chart", methods=['GET', 'POST'])
def chartPage():
    return render_template('chart.html')

def insert(email, val):
    try:
        cur = mysql.connection.cursor()
        s = str(datetime.datetime.now())[:10]
        cur.execute("Insert into dbts2(email, val, date) VALUES(%s, %s, %s)", (email,val, s))
        mysql.connection.commit()
        cur.close()
    except:
        return "failed"
    return "success"

@app.route("/visualize", methods = ['POST', 'GET'])
def drawPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        print(to_predict_dict)
        try:
            fig, ax = plt.subplots(figsize = (6, 6))
            ax = sns.set_style(style = "darkgrid")
            plt.ylim(0, 100)
            plt.axhline(55, linewidth = 1, color = 'red')
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM dbts2 where email = %s ", (to_predict_dict['eml'],))
            data = cur.fetchall()
            model = pickle.load(open("models/lr.pkl", 'rb'))
            l = []
            l1 = []
            time = []
            for i in data:
                l.append(model.predict([[float(i[2])]]))
                time.append(i[3])
            for i in l:
                l1.append(i[0][0])
            df = pd.DataFrame({'Date':time, 'Diabetes':l1})
            sns.lineplot(x = 'Date', y = 'Diabetes',ci = None, marker='o', data = df)

            canvas = FigureCanvas(fig)
            img = io.BytesIO()
            fig.savefig(img)
            img.seek(0)
            cur.close()
            return send_file(img, mimetype='img/png')
        except:
            message = "Invalid data"
            return render_template("home.html",message=message)



@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            if to_predict_dict['uname']:
                print(insert(to_predict_dict['uname'], to_predict_dict['Glucose']))
                del to_predict_dict['uname']
            print(to_predict_dict)
            for key, value in to_predict_dict.items():
                try:
                    if key == 'uname':
                        del to_predict_dict['uname']
                        break
                    else:
                        to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
            print(pred)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)



if __name__ == '__main__':
    app.run(debug = True)