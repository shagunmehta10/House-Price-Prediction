# import pandas as pd
# from flask import Flask, render_template, request
# import pickle 

# app = Flask(__name__)   # Use "app" consistently

# data=pd.read_csv('Cleaned_data.csv')

# pipe=pickle.load(open("RidgeModel.pkl",'rb'))

# @app.route('/')
# def index():

#     locations=sorted(data['location'].unique())
#     return render_template('index.html',locations=locations)

# @app.route('/predict',methods=['POST'])
# def predict():
#     location=request.form.get('location')
#     bhk=request.form.get('bhk')
#     bath=request.form.get('bath')
#     sqft=request.form.get('total_sqft')

#     print(location,bhk,bath,sqft)
#     input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_bhk','bath','bhk'])
#     prediction=pipe.predict(input)[0]
#     return str(prediction)




# if __name__ == "__main__":
#     app.run(debug=True, port=5001)


# import pandas as pd
# from flask import Flask, render_template, request
# import pickle 

# app = Flask(__name__)

# # Load dataset
# data = pd.read_csv('Cleaned_data.csv')

# # Load trained model
# pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
# with open("RidgeModel.pkl", "wb") as f:
#     pickle.dump(pipe, f)

# @app.route('/')
# def index():
#     locations = sorted(data['location'].unique())
#     return render_template('index.html', locations=locations)

# @app.route('/predict', methods=['POST'])
# def predict():
#     location = request.form.get('location')
#     bhk = request.form.get('bhk')
#     bath = request.form.get('bath')
#     sqft = request.form.get('total_sqft')

#     # Debug print (can be removed in production)
#     print(location, bhk, bath, sqft)

#     # Prepare input for prediction
#     input_df = pd.DataFrame(
#         [[location, sqft, bath, bhk]],
#         columns=['location', 'total_sqft', 'bath', 'bhk']
#     )

#     # Make prediction
#     prediction = pipe.predict(input_df)[0]

#     return str(round(prediction, 2))   # return rounded price

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)
 

import pandas as pd
from flask import Flask, render_template, request
import pickle 
import numpy as np

app = Flask(__name__)

# Load dataset
data = pd.read_csv('Cleaned_data.csv')

# Load trained model
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

import pickle
# After fitting your model as 'pipe'
with open("RidgeModel.pkl", "wb") as f:
    pickle.dump(pipe, f)

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)

    input_df = pd.DataFrame(
        [[location, sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    prediction = pipe.predict(input_df)[0]*1e5

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)