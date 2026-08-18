import pandas as pd
import pickle
from sklearn.linear_model import Ridge

# Load your cleaned data
data = pd.read_csv('Cleaned_data.csv')

# Prepare your features and target
X = data[['location', 'total_sqft', 'bath', 'bhk']]
y = data['price']

# If you have categorical features, use get_dummies or OneHotEncoder
X = pd.get_dummies(X, columns=['location'])

# Train Ridge regression model
pipe = Ridge()
pipe.fit(X, y)

# Save the trained model
with open("RidgeModel.pkl", "wb") as f:
    pickle.dump(pipe, f)