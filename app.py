from flask import Flask, jsonify, request

from functools import cmp_to_key

from dotenv import load_dotenv
load_dotenv()
import os

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from flask_cors import CORS, cross_origin

import numpy as np

db = SQLAlchemy()

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("sql_uri")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.app_context().push()
db.init_app(app)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

with app.app_context():
    db.reflect()

class City(db.Model):
    __table__ = db.metadata.tables["city"]
    def to_dict(self):
        return {field.name:getattr(self, field.name) for field in self.__table__.c}

class Weather(db.Model):
    __table__ = db.metadata.tables["weather"]
    def to_dict(self):
        return {field.name:getattr(self, field.name) for field in self.__table__.c}
    
class Tax(db.Model):
    __table__ = db.metadata.tables["tax"]
    def to_dict(self):
        return {field.name:getattr(self, field.name) for field in self.__table__.c}
    
class IncomeTax(db.Model):
    __table__ = db.metadata.tables["incometax"]
    def to_dict(self):
        return {field.name:getattr(self, field.name) for field in self.__table__.c}

def makeMeanDeviation(arr, key):
    total = 0
    for r in arr:
        total += r[key]
    mean = total/len(arr)
    x = 0
    for r in arr:
        x += (r[key]-mean)**2
    stdeviation = (x/len(arr))**0.5
    return mean, stdeviation

def cosine_similarity_numpy(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)

data = db.session.query(City,Weather,Tax).join(Weather).order_by(City.id).join(Tax)
cities = [r[0].to_dict() for r in data.all()]
weather = [r[1].to_dict() for r in data.all()]
tax = [r[2].to_dict() for r in data.all()]

keys = ["julytemp","salestax", "density", "growth", "propertytaxquarter",
        "julyhumidity", "janprecipitation", "julyprecipitation"]

vectors = [[] for city in cities]
for i in keys:
    cur_array = []
    if i in cities[0]:
        cur_array = cities
    elif i in weather[0]:
        cur_array = weather
    else:
        cur_array = tax

    mean, standard = makeMeanDeviation(cur_array,i)

    for j in range(0,len(cur_array)):
        vectors[j].append((cur_array[j][i]-mean)/standard)


@app.route('/')
def hello():
    return jsonify(cosine_similarity_numpy([0.5,0.5],vectors[1]))

@app.route('/search', methods=['POST'])
def search():
    # This would take in quiz answers as they have been converted to values in same format as DB
    data = request.get_json()
    user_vector = []
    print(data)
    for key in keys:
        
        if key in data:
            cur_array = []
            if key in cities[0]:
                cur_array = cities
            elif key in weather[0]:
                cur_array = weather
            else:
                cur_array = tax

            mean, standard = makeMeanDeviation(cur_array,key)
            value = mean+(float(data[key])*standard)
            user_vector.append((value-mean)/standard)
        else:
            user_vector.append(0)
    max_val = -1
    max_id = 1
    def compare (f1,f2):
        return cosine_similarity_numpy(user_vector,f1["vector"]) - cosine_similarity_numpy(user_vector,f2["vector"])
    temp_vectors = []
    for i in range(0,len(vectors)):
        temp_vectors.append({"id":i,"vector":vectors[i]})
    temp_vectors = sorted(temp_vectors, key=cmp_to_key(compare), reverse=True)
    
    return jsonify(temp_vectors[:4])



#TODO: Structure Vector Search
# 1. Connect to the DB and fetch all city entries
# 2. Construct VECTOR DB 
# 3. Open API which takes in quiz answers and converts them to vector
# 4. Conduct vector searches