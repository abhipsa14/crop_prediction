from flask import Flask, render_template,request,url_for
import joblib 
import pandas as pd
from pymongo import MongoClient


std_scaler = joblib.load('./models/std_scaler.lb')
kmeans_model = joblib.load('./models/kmeans_model.lb')
df = pd.read_csv("./models/filter_crops.csv")
app=Flask(__name__)

connection_string="mongodb+srv://abhipsasri8183:fccv5v9jXuJIs4W6@cluster1.b94xm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client=MongoClient(connection_string)
database=client['Farmer'] #-->database
collection=database["FarmerData"] #table create or collection


@app.route('/')
def home():
   return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def  predict(): 
    if(request.method=='POST'):
        N=int(request.form['N'])
        P=int(request.form['P'])
        K=int(request.form['K'])
        humidity=float(request.form['humidity'])
        temperature=float(request.form['temperature'])
        ph=float(request.form['ph'])
        rainfall=float(request.form['rainfall'])
        UNSEEN_DATA=[[N,P,K,temperature,humidity,ph,rainfall]]
        transformed_data=std_scaler.transform(UNSEEN_DATA) #standard distributed
        cluster=kmeans_model.predict(transformed_data)[0] #single dimensin value aayi thi
        suggestion_crops=list(df[df['cluster_no']==cluster]['label'].unique())
        data={"N":N,"P":P,"K":K,"temperature":temperature,"humidity":humidity,"ph":ph,"rainfall":rainfall}

        data_id=collection.insert_one(data).inserted_id
        print("Your data is inserted into the mongodb your record id is:",data_id)
        return render_template('home.html',pred=str(suggestion_crops))

#Flask Framework returns an array always.




if __name__=="__main__":
    app.run(debug=True)