
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import pickle

# === 1) Load data ===
df = pd.read_csv('Cleaned_data.csv')  # Ensure this file is in the same folder

required_cols = ['location', 'total_sqft', 'bath', 'bhk', 'price']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in Cleaned_data.csv: {missing}")

X = df[['location', 'total_sqft', 'bath', 'bhk']]
y = df['price']  # assumed to be in lakhs

# === 2) Build preprocessing + model pipeline ===
numeric_features = ['total_sqft', 'bath', 'bhk']
categorical_features = ['location']

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

model = Ridge()

pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])

# === 3) Fit ===
pipe.fit(X, y)

# === 4) Save trained pipeline ===
with open('RidgeModel.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Also save the locations used during training (optional, helpful for UI)
enc = pipe.named_steps['preprocess'].named_transformers_['cat']
locations = sorted(list(enc.categories_[0]))
pd.Series(locations).to_csv('locations.csv', index=False, header=False)

print('✅ Saved RidgeModel.pkl and locations.csv')
