import sys
sys.path.append("/usr/local/lib/python3.9/dist-packages/")
sys.path.append("/usr/lib/python3/dist-packages/")
from flask import Flask
from flask import jsonify, request
from flask_pymongo import PyMongo
from bson.json_util import dumps
from bson.objectid import ObjectId
import json



app = Flask(__name__)

app.secret_key = "secretKey"
app.config["MONGO_URI"] = "mongodb://localhost:27017/flask"

mongo = PyMongo(app)

@app.route('/add', methods=["POST"])
def addUser():
    _json = request.json
    print(_json["name"])

    id = mongo.db.student_data.insert_one({"gre_score": _json["name"], "toefl_score": _json["age"]})

    print(id)

    res = jsonify("success?")

    return res

@app.route('/all', methods=["GET"])
def getAlldocs():
    
    data = mongo.db.student_data.find()

    return dumps(data)

if __name__ == "__main__":
    print("hello world")
    app.run(debug=True)
    

