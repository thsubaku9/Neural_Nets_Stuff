#mysom

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

from minisom import MiniSom

som = MiniSom(x = 20, y = 20, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(X)

threshold = 0.5

myData = list()

distance = som.distance_map().T

for i in range(distance.shape[0]):
    avg_dist = distance[i].sum()/distance[i].shape
    if (avg_dist >= threshold):
        j = distance[i].argmax()

        if(len(mappings[(i,j)]) > 0):
            for x in mappings[(i,j)]:
                myData.append(x)

myData = np.array(myData)
#12,10 and 9,18 are chosen on the basis of high IND values, will differ based on MiniSom size
frauds = np.concatenate((myData),axis = 0)
frauds = frauds.reshape(myData.shape)
frauds = sc.inverse_transform(frauds)


#Unsuper -> super
customers = dataset.iloc[:, 1:].values
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if np.intersect1d(dataset.iloc[i,0],frauds).size >0 :
        is_fraud[i] = 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

from keras.layers import Dense,Dropout
from keras.models import Input,Model
from keras import losses,optimizers


feedIn = Input(shape = (customers.shape[1],))

layer_1 = Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu')(feedIn)
output_layer = Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')(layer_1)
model = Model(feedIn,output_layer)
model.compile(optimizer = optimizers.Adam(), loss = losses.binary_crossentropy, metrics = ['accuracy'])
model.fit(customers, is_fraud, batch_size = 2, epochs = 3)

y_pred = model.predict(customers)
c_loss = y_pred - is_fraud.reshape((690,1))
a_error = c_loss.sum() / c_loss.shape[0]
print("Average prediciton error:{} ".format(a_error))

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]
