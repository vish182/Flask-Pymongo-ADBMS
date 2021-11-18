import sys
# sys.path.append("/usr/local/lib/python3.9/dist-packages/")
# sys.path.append("/usr/lib/python3/dist-packages/")
# sys.path.append("/home/vish182/.local/lib/python3.9/site-packages")
from flask import Flask
from flask import jsonify, request
from flask_pymongo import PyMongo
from bson.json_util import dumps
from bson.objectid import ObjectId
import json
from flask_cors import CORS

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error

import pickle

df= pd.read_csv("Admission_Predict.csv")
df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})

x = df.iloc[:, 1:8]
y = df.iloc[:, -1]

linear_regressor = LinearRegression()
linear_regressor.fit(x,y)

pickle.dump(linear_regressor,open('linear_regressor.pkl2','wb'))

model = pickle.load(open('linear_regressor.pkl2','rb'))


app = Flask(__name__)
CORS(app)

app.secret_key = "secretKey"
app.config["MONGO_URI"] = "mongodb://localhost:27017/flask"

mongo = PyMongo(app)



@app.route('/add', methods=["POST"])
def addUser():
    _json = request.json
    print(_json["name"])
    print("please")

    #id = mongo.db.student_data.insert_one({"gre_score": _json["name"], "toefl_score": _json["age"]})

    #print(id)

    res = jsonify("success?")

    return res

# to use the predict button in the web app
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Chance of Admission = {}'.format(output))

def extractAttributes(doc):
    d = {"gre": int(doc["gre"]), "toefl": int(doc["toefl"]), "uniRating": int(doc["uniRating"]), "sop": int(doc["sop"]), "lor": int(doc["lor"]), "research": bool(doc["research"]), "cgpa": float(doc["cgpa"])}
    return d


@app.route('/getPrediction', methods=["POST"])
def getPrediction123():
    _json = request.json
    print(_json)
    print("please")

    processed = extractAttributes(_json)

    id = mongo.db.student_data.insert_one(processed)
    
    int_features = [processed["gre"], processed["toefl"], processed["uniRating"], processed["sop"], processed["lor"], processed["cgpa"], processed["research"]]
    final_features = [np.array(int_features)]
    print("final features: ", final_features)
    prediction = model.predict(final_features)
    print("predixtion: ", prediction)

    output = round(prediction[0], 2)
    print("round: ", output)
    #print(id)
    res = jsonify(output)
    return res

@app.route('/all', methods=["GET"])
def getAlldocs():
    
    data = mongo.db.student_data.find()

    return dumps(data)

if __name__ == "__main__":
    print("hello world")
    app.run(debug=True)
    

