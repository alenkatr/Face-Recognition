#VSE SKUPAJ-----CV verzija 29.3.2021---------------IMAGES--------------------basic kNN with euclidean metric-----------------------------
#To add new metric: https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
#deluje----ok
#1) load data IMAGES	2) defin PM	4) lmnn	4) knn	5) CV


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score   #for crossvalidation
from sklearn.model_selection import StratifiedKFold

from pylmnn import LargeMarginNearestNeighbor as LMNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score    #for confusion matrix
from sklearn import datasets, neighbors, metrics   # Import datasets, classifiers and performance metrics

from sklearn.preprocessing import Normalizer	#for normalization

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler    ##for standardization-Gaussian with zero mean and unit variance.


# X: the data to fit. Can be for example a list, or an array. y: the target variable to try to predict in the case of supervised learning.
#for iris dataset: X, y = datasets.load_iris(return_X_y=True)  	
#X, y = datasets.load_digits(return_X_y=True)
#print(datasets.load_digits(return_X_y=True))
#X,y = datasets.load_digits(return_X_y=True)
#X, y = datasets.load_iris(return_X_y=True)		#X--data, y--class

train_df = pd.read_csv('japanBpp.csv')
X = np.array(train_df.iloc[:, :-1])	#all rows & all column except last one
#print(train_data)
y = np.array(train_df.iloc[:, -1])   #all rows & only last column

X.shape
print(f"Mn X {X.shape}")

#Normalization dataset
transformer = Normalizer().fit(X)
X = transformer.transform(X)


#print(X)
#print(y)
#Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated = standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# LMNN: algorithm for metric learning---------------------after cross validation
# Set up the hyperparameters
k_train, k_test, max_iter = 3, 3, 2000	#n_components=X.shape[1], za bpp:640

# Instantiate the metric learner
lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter)

# Train the metric learner
lmnn.fit(X,y)

# Create a classifier: a knn with PM
for i in range(1,11):
	knn = neighbors.KNeighborsClassifier(n_neighbors=i, leaf_size=3, algorithm='auto', metric='euclidean', n_jobs=4) 	#to add new metric: metric='pyfunc'
	#for testing
	#knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='euclidean')    #algorithm='ball_tree' - for a custom metric

	# Learn the digits/iris on the train subset - fit the model
	knn.fit(lmnn.transform(X),y)

	#cross validation-------------------where is this------------after knn
	scores = cross_val_score(knn, X, y, cv=StratifiedKFold(10), n_jobs=4)	#rez. se ne spremeijo, če dam tuki lmnn.transform(X)



	# Predict the value of the digit on the test subset
	predicted = knn.predict(lmnn.transform(X))

	#for output in txt file
	with open("outputjapanBPPEM.txt", "a") as f:
		print(f"Classification report for classifier {knn}:\n"
		      f"{metrics.classification_report(y, predicted)}\n", file=f)   
		print('Name of data:', file=f)

		# Print out confusion matrix
		cmat = confusion_matrix(y, predicted)	#print(cmat)

		print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))), file=f)
		print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))), file=f)

		print(cmat, file=f)

		print('Diagonal (sum): ', np.trace(cmat), file=f)
		print("All Instances:", np.sum(cmat), file=f)
		print("Correctly Classified Instances:", np.trace(cmat)/np.sum(cmat), file=f)
		print('Number of neighbors: ', i, file=f)



