import sklearn.model_selection as sk
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  

print ('feature extraction')

YouTubeActions_TotalFeatures = sio.loadmat('Encoded Features/Encoded_features_Youtube.mat')
YouTubeActions_TotalFeatures = YouTubeActions_TotalFeatures['Encoded_features'] #for converting into python 2D array

DatabaseLabels = sio.loadmat('Encoded Features/Labels.mat')
DatabaseLabels = DatabaseLabels['labels']
DatabaseLabels = DatabaseLabels.reshape(6437)




X_Train, X_Test, Y_Train , Y_Test = sk.train_test_split(YouTubeActions_TotalFeatures, DatabaseLabels, test_size=0.20,random_state = 42 ) #, shuffle=False
print (X_Train.shape, ',', Y_Train.shape , ' , ', Y_Train[0])

#X_Train, Y_Train , X_Validation , Y_Validation = sk.train_test_split(X_Train,Y_Train,test_size=0.20,random_state = 42 ) #, shuffle=False
print ('Training')
#svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='rbf')
svclassifier = SVC(kernel='poly')

svclassifier.fit(X_Train, Y_Train)
print ('Predictions')
y_pred = svclassifier.predict(X_Test)  
 
print(confusion_matrix(Y_Test,y_pred))  
print(classification_report(Y_Test,y_pred)) 



print ('done')