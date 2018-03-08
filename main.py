# %% Load up the data:

import numpy
from NeuralNet import NeuralNet
from mnist import MNIST

mndata = MNIST( 'data' )
mndata.gz = True

trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_training()

trainSetImages = dict()
trainSetLabels = dict()

testSetImages = dict()
testSetLabels = dict()

for num, im, label in zip( range( len( trainingImages ) ), trainingImages, trainingLabels ):
	# feature scaling is important
	trainSetImages[ num ] = numpy.array( im ) / 255.0
	trainSetLabels[ num ] = label


for num, im, label in zip( range( len( testImages ) ), testImages, testLabels ):
	testSetImages[ num ] = numpy.array( im ) / 255.0
	testSetLabels[ num ] = label

print( 'Data loaded.' )

# %% Build the neural network based on the configured parameters:

nn = NeuralNet( inputDim=784, hiddenDim=100, outputDim=10 )

nn.train( trainSetImages, trainSetLabels, 100 )

# print( nn.predict( trainingImages[ 0 ] ) )
