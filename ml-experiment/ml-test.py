import joblib
import numpy as np

filename = 'logistic_model.joblib'
loaded_model = joblib.load(filename) # rb means read as binary
x_test = joblib.load("X_test.joblib")
print(x_test[10])
result = loaded_model.predict(x_test)
print(result)