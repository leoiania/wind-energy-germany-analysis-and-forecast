from flask import Flask,request,render_template
import sys
import numpy as np


from src.exception import CustomException
from src.utils import load_object
from src.pipeline.predict_pipeline import PredictPipeline,CustomData


app = Flask(__name__)

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(forecast_days = request.form.get('forecast_days'))

        
        model = PredictPipeline()
        days=data.get_data_web()
        results = model.predict(days = days)
        results = [np.round(i,3) for i in results]

        return render_template('home.html',results = results)
        

if __name__=="__main__":
    app.run(debug='True',host="0.0.0.0")     