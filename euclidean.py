import pickle
import argparse
import numpy

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=True,
	help="path test dataset pickle")
ap.add_argument("-l", "--learning", required=True,
	help="path to training dataset pickle")

args = vars(ap.parse_args())

train_dict=pickle.loads(open(args['learning'], "rb").read())
test_dict=pickle.loads(open(args['test'], "rb").read())

class AccuracyTester:
	def __init__(self, test_dict, train_dict):
		self.test_dict=test_dict
		self.train_dict=train_dict
	def calculate_minimum_distance(self, tested_encoding):
		distances=[]
		for known_encoding in self.train_dict['encodings']:
			distance=numpy.linalg.norm(known_encoding-tested_encoding)
			distances.append(distance)
		min_index=distances.index(min(distances))
		name=self.train_dict['names'][min_index]
		return((name, distances[min_index]))
		# print('The most similar person is person {} with distance {}.'.format(name, min(distances)))
	def test_accuracy(self):
		total=len(test_dict['encodings'])
		successes=0.
		for encoding, name in zip(test_dict['encodings'], test_dict['names']):
			pair=self.calculate_minimum_distance(encoding)
			# print('Calculated label is {}, real label is {} with distance {}'.format(pair[0], name, pair[1]))
			if int(pair[0])==int(name):
				successes+=1
		print('Accuracy: '+str(100*successes/total)+'%')

tester=AccuracyTester(test_dict, train_dict)
tester.test_accuracy()
