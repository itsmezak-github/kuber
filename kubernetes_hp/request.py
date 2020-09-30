import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'bedrooms':3, 'bathrooms':1, 'sqft_lot':5650, 'floors':1, 'condition':2, 'yr_built':1900})

print(r.json())
