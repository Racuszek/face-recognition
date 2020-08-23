import pickle
import argparse
import numpy

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=True,
	help="path test dataset pickle")
ap.add_argument("-l", "--learning", required=True,
	help="path to training dataset pickle")
ap.add_argument("-k", "--kvalue", required=True,
	help="number of nearest neighbours")

args = vars(ap.parse_args())

train_dict=pickle.loads(open(args['learning'], "rb").read())
'''
Dictionary containing training data in the following format:
{ 'names': ['Adam', 'Barbara', 'Cindy'],
  'encodings': [[0.123, -0.456, 0.321], [-0.990, 0.300, -1.432], [2.513, -0.100, 2.102]]}
'''
test_dict=pickle.loads(open(args['test'], "rb").read())
# Dictionary containing test data in the same format
k=int(args['kvalue'])

class AccuracyTester:
	def __init__(self, test_dict, train_dict, k):
		self.test_dict=test_dict
		self.train_dict=train_dict
		self.k=k
	def calculate(self, tested_encoding):
		# Array for storing tuples: (name, distance)
		pairs=[]
		
		i=0
		for known_encoding in self.train_dict['encodings']:
			this_distance=numpy.linalg.norm(known_encoding-tested_encoding)
			this_name=self.train_dict['names'][i]
			pairs.append((this_name, this_distance))
			i=i+1
		# Sorting the array by the distances in ascending order
		pairs.sort(key=lambda x: x[1])
		# Extracting k most probable names
		k_names=[pair[0] for pair in pairs[:self.k]]
		# Looking for the most common name among k names
		counts=numpy.bincount(k_names)
		most_common_name=numpy.argmax(counts) # Results for k=2 are better than k=1 – why, how does this function work?
		return(most_common_name)
	def test_accuracy(self):
		# Maximum number of points is equal for the length of the encodings list
		total=len(test_dict['encodings'])
		successes=0.
		for encoding, name in zip(test_dict['encodings'], test_dict['names']):
			guess=self.calculate(encoding)
			# Add a point for each matching label
			if guess==int(name):
				successes+=1
		print('Accuracy: '+str(100*successes/total)+'%')

tester=AccuracyTester(test_dict, train_dict, k)
tester.test_accuracy()