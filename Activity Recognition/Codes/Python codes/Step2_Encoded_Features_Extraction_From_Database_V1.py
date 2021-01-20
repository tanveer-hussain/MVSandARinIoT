from keras.models import load_model  # Will be used to retrieve the saved weights of the trained NN.
import sklearn.model_selection as sk
import scipy.io as sio
import numpy as np

encoder_size = 4000
encoder = load_model(r'./weights/encoder_weights.h5')  # Load the TRAINED encoder.
#decoder = load_model(r'./weights/decoder_weights.h5')  # Load the TRAINED decoder.

#loading data
x_train = sio.loadmat('X_Train.mat')
x_train = x_train['X_Train']

x_test = sio.loadmat('X_Test.mat')
x_test = x_test['X_Test']

x_val = sio.loadmat('X_Validation.mat')
x_val = x_val['X_Validation']

#Training features extraction

train_len = x_train.shape
x_train_features = np.zeros((train_len[0], encoder_size))
for i in range(0, train_len[0]-1):
    f1 = x_train[i][:]
    f1 = f1.reshape(1,15000)
    inputs = np.array(f1)
    print ('Training features ' , i , ' of ', train_len[0])
    single_features = encoder.predict(inputs)
    x_train_features[i][:] = single_features
print ('Storing train features shape = ', x_train_features.shape)
Encoded_train_features = {}
Encoded_train_features['Encoded_train_features'] = x_train_features
sio.savemat('Encoded Features/Encoded_train_features.mat',Encoded_train_features)
print ('Train Features Saved')


#Testing features extraction
test_len = x_test.shape
x_test_features = np.zeros((test_len[0], encoder_size))
for i in range(0, test_len[0]-1):
    f1 = x_test[i][:]
    f1 = f1.reshape(1,15000)
    inputs = np.array(f1)
    print ('Testing features ' , i , ' of ', test_len[0])
    single_features = encoder.predict(inputs)
    x_test_features[i][:] = single_features
print ('Storing test features shape = ', x_test_features.shape)
Encoded_test_features = {}
Encoded_test_features['Encoded_test_features'] = x_test_features
sio.savemat('Encoded Features/Encoded_test_features.mat',Encoded_test_features)
print ('Test Features Saved')

#Validation features extraction
val_len = x_val.shape
x_val_features = np.zeros((val_len[0], encoder_size))
for i in range(0, val_len[0]-1):
    f1 = x_val[i][:]
    f1 = f1.reshape(1,15000)
    inputs = np.array(f1)
    print ('Validation features ' , i , ' of ', val_len[0])
    single_features = encoder.predict(inputs)
    x_val_features[i][:] = single_features
print ('Storing validation features shape = ', x_test_features.shape)
Encoded_val_features = {}
Encoded_val_features['Encoded_val_features'] = x_val_features
sio.savemat('Encoded Features/Encoded_val_features.mat',Encoded_val_features)
print ('Validation Features Saved')


print ('done')