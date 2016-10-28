import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

#load data from data.csv
#average temperature from http://idojarasbudapest.hu/archivalt-idojaras - (min+max)/2
data = np.loadtxt("data.csv", dtype=float, delimiter=",")
data_input = []
data_output = []

#collect from data for inputs
#we use 30 tempreature to predict the value on the day, week and month after them
#data is in descending order, so i means one day, i-7 means one week and i-30 means one month before
for i in range(30, len(data)-30):
    data_input.append(data[i + 1:i + 31, 3])
    data_output.append(np.array([data[i, 3], data[i-7, 3], data[i-30, 3]]))

X = np.array(data_input)
Y = np.array(data_output)

#scale the inputs
scaler = preprocessing.StandardScaler().fit(X)
joblib.dump(scaler, 'scaler.pkl')
X = scaler.transform(X)

input_dim = X.shape[1]
output_dim = Y.shape[1]

#we use a fully connected network
model = Sequential()
model.add(Dense(input_dim=input_dim, output_dim=128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
for i in range(6):
    model.add(Dense(input_dim=128, output_dim=128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
model.add(Dense(input_dim=128, output_dim=output_dim))
model.add(Activation('relu'))

model.compile(loss='mse', optimizer='rmsprop', metrics=["mean_squared_error"])

print("Training...")
model.fit(X, Y, nb_epoch=2000, batch_size=25, validation_split=0.1,  verbose=0, shuffle=True)
model.save('weather.h5')