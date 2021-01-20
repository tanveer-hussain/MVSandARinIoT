from keras.layers import Input, Dense
from keras.models import Model

import sklearn.model_selection as sk
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt

x_train = sio.loadmat('Youtube features/X_Train.mat')
x_train = x_train['X_Train']

x_test = sio.loadmat('Youtube features/X_Test.mat')
x_test = x_test['X_Test']

x_val = sio.loadmat('Youtube features/X_Validation.mat')
x_val = x_val['X_Validation']



print ("X train = " , x_train.shape)
print ("X Val = " , x_val.shape)
print ("X test = " , x_test.shape)


input_dim = Input(shape = (15000, ))
epoch = 1000
# Encoder Layers
encoded1 = Dense(8000, activation = 'relu')(input_dim)
encoded2 = Dense(1000, activation = 'relu')(encoded1)

encoder = Model(input_dim, encoded2)
encoder.save(r'./weights/encoder_weights.h5')
# Decoder Layers
decoded1 = Dense(8000, activation = 'relu')(encoded2)
decoded2 = Dense(15000, activation = 'relu')(decoded1)

# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded2)

autoencoder.save(r'./weights/ae_weights.h5')




#decoder.save(r'./weights/decoder_weights.h5')

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(x_train, x_train,
                epochs=epoch,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val, x_val))



loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = range(epoch)
plt.figure()
plt.plot(epoch, loss, 'bo', label='Training loss')
plt.plot(epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print ('done')