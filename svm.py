from sklearn import svm
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=True,
	help="path test dataset pickle")
ap.add_argument("-l", "--learning", required=True,
	help="path to training dataset pickle")

args = vars(ap.parse_args())

# Opening the passed pickles
train_dict=pickle.loads(open(args['learning'], "rb").read())
test_dict=pickle.loads(open(args['test'], "rb").read())

# Creating and training the classifier
clf=svm.SVC(gamma=3, C=100.)
clf.fit(train_dict['encodings'], train_dict['names'])

# Calculating the classifier accuracy of test data
arr_of_predictions=clf.predict(test_dict['encodings'])
total=len(arr_of_predictions)
score=0.
for pair in zip(arr_of_predictions, test_dict['names']):
	if(pair[0]==pair[1]):
		score+=1
print('Accuracy: ', 100*score/total, '%')

# print(clf.predict(test_dict['encodings'][-1:]))
# print('true value: ', test_dict['names'][-1])