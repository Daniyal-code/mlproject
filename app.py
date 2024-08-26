from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData



application = Flask(__name__)

app = application

# route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        # if its a post req, we will create data
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score =float(request.form.get('writing_score')),
            reading_score=float(request.form.get('reading_score'))

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results= predict_pipeline.predict(pred_df) #once this func is called, first transformation will happen on the given dataframe, then prediction will be returned
        print(results)
        return render_template('home.html',results = results[0])
        # 
    # now read this results value to home.html


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
