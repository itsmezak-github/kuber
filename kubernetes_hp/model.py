# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


from sklearn import preprocessing
data = pd.read_csv('house_data.csv')
df = data.copy()

from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')

labels =df['price']
#conv_dates = [1 if values ==2014 else 0 for values in df.date]
#df['date'] = conv_dates
train1 = df.drop(['price'],axis=1)

#Splitting Training and Test Set
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
#reg = LinearRegression(normalize=False,fit_intercept=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10, random_state=2)


#Fitting model with trainig data
from sklearn import linear_model
#reg = LinearRegression()
#reg.fit(x_train,y_train)

from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 4, min_samples_split = 2,
         learning_rate = 0.1, loss = 'ls')
clf.fit(x_train, y_train)


#reg = MLPRegressor(random_state=1, max_iter=2000).fit(x_train, y_train)
#print(reg.score(x_test,y_test))
print(clf.score(x_test,y_test))


# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[3, 1, 5650, 1]]))
