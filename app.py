from flask import Flask,render_template, url_for,request
import pandas as pd
import pickle

model = pickle.load(open('Random_forest_regressor_pkl' , 'rb'))
app =Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods =['POST'])
def predict():
    df = pd.read_csv('real_2018.csv')
    predictions = model.predict(df.iloc[:,:-1].values)
    predictions = predictions.tolist()
    return render_template('result.html',prediction = predictions)

if __name__ == '__main__':
    app.run(debug=True)