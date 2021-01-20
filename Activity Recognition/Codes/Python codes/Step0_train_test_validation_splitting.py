import sklearn.model_selection as sk
import scipy.io as sio



YouTubeActions_TotalFeatures = sio.loadmat('YouTube_Actions_Features.mat')
YouTubeActions_TotalFeatures = YouTubeActions_TotalFeatures['TotalFeatures'] #for converting into python 2D array


X_Train, X_Test = sk.train_test_split(YouTubeActions_TotalFeatures,test_size=0.20,random_state = 42 ) #, shuffle=False

X_Train, X_Validation = sk.train_test_split(X_Train,test_size=0.20,random_state = 42 ) #, shuffle=False

print (X_Train.shape, ',', X_Test.shape)


Train_data = {}
Test_data = {}
Validation_data = {}

Train_data['X_Train'] = X_Train
Test_data['X_Test'] = X_Test
Validation_data['X_Validation'] = X_Validation

sio.savemat('X_Train.mat',Train_data)
sio.savemat('X_Test.mat',Test_data)
sio.savemat('X_Validation.mat',Validation_data)
print ('Done')


#sio.savemat('Complete_Dataset.mat', dictt )
