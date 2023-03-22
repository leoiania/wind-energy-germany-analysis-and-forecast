from flask import Flask,request,render_template
import sys


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

        return render_template('home.html',results = results)
        
# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route('/')
# def index():
#     return render_template('index.html')

if __name__=="__main__":
    app.run(debug='True',host="0.0.0.0")     