import numpy as np
import scipy
import scipy.io
import argparse
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ld', type=float, default=500000) # lambda
	return parser.parse_args()


def normalizeFeature(x):
	# x = d x N dims (d: feature dimension, N: the number of features)
	x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
	feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	feat = x / feature_norm[:, np.newaxis]
	return feat

def SAE(x, s, ld):
	# SAE is Semantic Autoencoder
	# INPUTS:
	# 	x: d x N data matrix
	#	s: k x N semantic matrix
	#	ld: lambda for regularization parameter
	#
	# OUTPUT:
	#	w: kxd projection matrix

	A = np.dot(s, s.transpose())
	B = ld * np.dot(x, x.transpose())
	C = (1+ld) * np.dot(s, x.transpose())
	w = scipy.linalg.solve_sylvester(A,B,C)
	print(w.shape)
	return w

def distCosine(x, y):
	xx = np.sum(x**2, axis=1)**0.5
	x = x / xx[:, np.newaxis]
	yy = np.sum(y**2, axis=1)**0.5
	y = y / yy[:, np.newaxis]
	dist = 1 - np.dot(x, y.transpose())
	return dist



def zsl_acc(semantic_predicted, semantic_gt, opts):
	# zsl_acc calculates zero-shot classification accruacy
	#
	# INPUTS:
	#	semantic_prediced: predicted semantic labels
	# 	semantic_gt: ground truth semantic labels
	# 	opts: other parameters
	#
	# OUTPUT:
	# 	zsl_accuracy: zero-shot classification accuracy (per-sample)

	dist = 1 - distCosine(semantic_predicted, normalizeFeature(semantic_gt.transpose()).transpose())
	y_hit_k = np.zeros((dist.shape[0], opts.HITK))
	for idx in range(0, dist.shape[0]):
		sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
		y_hit_k[idx,:] = opts.test_classes_id[sorted_id[0:opts.HITK]]

	n = 0
	for idx in range(0, dist.shape[0]):
		if opts.test_labels[idx] in y_hit_k[idx,:]:
			n = n + 1
	zsl_accuracy = float(n) / dist.shape[0] * 100
	return zsl_accuracy, y_hit_k


def main():
	# for AwA dataset: Perfectly works.
	'''
	AWA data details:
	X_tr: (24295, 1024)
	X_te: (6108, 1024)
	S_tr: (24295, 85)
	test_labels: (6180,)
	testclasses_id:(10,)
	S_te_gt: (10, 85)
	'''

	'''
	# AWA
	opts = parse_args()
	awa = scipy.io.loadmat('../data/awa_demo_data.mat')
	train_data = awa['X_tr']
	test_data = awa['X_te']
	train_class_attributes_labels_continuous_allset = awa['S_tr']
	opts.test_labels = awa['test_labels']
	opts.test_classes_id = awa['testclasses_id']
	test_class_attributes_labels_continuous = awa['S_te_gt']
	'''

	# AWA2_test_attributlabel: (6985, 85)
	# AWA2_test_continuous_01_attributelabel: (6985, 85)
	# AWA2_testlabel: (6985,) -> test_labels
	# AWA2_train_continuous_01_attributelabel: (30337, 85) -> S_tr
	# AWA2_trainlabel: (30337,)
	# resnet101_testfeatures: (6985, 2048) -> X_te
	# resnet101_trainfeatures: (30337, 2048) -> X_tr
	# AWA2_train_attributelabel: (30337, 85)

	opts = parse_args()
	
	# food 11 dataset:
	train_data = np.load('../food11/resnet101_trainfeatures.npy')
	test_data = np.load('../food11/resnet101_testfeatures.npy')
	train_class_attributes_labels_continuous_allset = np.load('../food11/Food11_train_Label_attributelabel.npy')
	opts.test_labels = np.load('../food11/Food11_testlabel.npy')
	test_classes_id_list = [0, 5, 9]
	opts.test_classes_id = np.asarray(test_classes_id_list)
	test_class_attributes_labels_continuous = np.load('../food11/S_te_gt_food_Label_normed.npy')
	
	'''
	# AWA2 attribute dataset:
	train_data = np.load('../AwA2/resnet101_trainfeatures.npy')
	test_data = np.load('../AwA2/resnet101_testfeatures.npy')
	train_class_attributes_labels_continuous_allset = np.load('../AwA2/AWA2_train_continuous_01_attributelabel.npy')
	opts.test_labels = np.load('../AwA2/AWA2_testlabel.npy')
	test_classes_id_list = [5, 13, 14, 17, 23, 24, 33, 38, 41, 47]
	opts.test_classes_id = np.asarray(test_classes_id_list)
	test_class_attributes_labels_continuous = np.load('../AwA2/S_te_gt.npy')
	##### Normalize the data
	train_data = normalizeFeature(train_data.transpose()).transpose() 
	'''
	##### Training
	# SAE
	W = SAE(train_data.transpose(), train_class_attributes_labels_continuous_allset.transpose(), opts.ld) 

	##### Test
	opts.HITK = 1
	
	# [F --> S], projecting data from feature space to semantic space: 84.68% for AwA dataset
	semantic_predicted = np.dot(test_data, normalizeFeature(W).transpose())
	[zsl_accuracy, y_hit_k] = zsl_acc(semantic_predicted, test_class_attributes_labels_continuous, opts)
	print('[1] zsl accuracy for AwA dataset [F >>> S]: {:.2f}%'.format(zsl_accuracy))

	# [S --> F], projecting from semantic to visual space: 84.00% for AwA dataset
	test_predicted = np.dot(normalizeFeature(test_class_attributes_labels_continuous.transpose()).transpose(), normalizeFeature(W))
	[zsl_accuracy, y_hit_k] = zsl_acc(test_data, test_predicted, opts)
	print('[2] zsl accuracy for AwA dataset [S >>> F]: {:.2f}%'.format(zsl_accuracy))
	
if __name__ == '__main__':
	main()
