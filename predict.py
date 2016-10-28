import datetime
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model
data = np.loadtxt("data.csv", dtype=float, delimiter=",")

#to get a prediction, you have to fill up the data table until today
#from those data we will get the temperature for tomorrow, next week, and next month
scaler = joblib.load('scaler.pkl')
X = np.array([data[0:30, 3]])
X = scaler.transform(X)

model = load_model('weather.h5')
preds = model.predict(X, verbose=0)

date = datetime.date(int(data[0][0]), int(data[0][1]), int(data[0][2]))
print "Tomorrow (", date + datetime.timedelta(days=1), "): ",  preds[0,0]
print "Next week (", date + datetime.timedelta(days=7), "): ",  preds[0,1]
print "Next month (", date + datetime.timedelta(days=30), "): ",  preds[0,2]
