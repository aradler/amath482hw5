import numpy
from typing import List
from random import randint


class NeuralNet:
	def __init__(
		self,
		inputDim: int=729,
		hiddenDim: int=100,
		outputDim: int=10 ):

		self.inputDim = inputDim
		self.hiddenDim = hiddenDim
		self.outputDim = outputDim

		self.layers = [
			numpy.random.rand( inputDim, hiddenDim ),
			numpy.random.rand( hiddenDim, outputDim ) ]

		self.layerOutputs = []
		self.activations = []

	def train(
		self,
		trainSetImages: List[numpy.array],
		trainSetLabels: List[int],
		numIter: int,
		gradientStep: float=0.0001 ):

		for i in range( numIter ):
			print( 'Training iteration {}'.format( i ) )
			thisSampleNum = randint( 0, len( trainSetImages ) )

			trainImage = trainSetImages[ thisSampleNum ]
			trainLabel = trainSetLabels[ thisSampleNum ]

			self.__trainStep( trainImage, trainLabel, gradientStep )
		pass

	def __trainStep(
		self,
		trainImage: numpy.array,
		trainLabel: int,
		gradientStep: float=0.0001 ):
		"""
		Does a single step of training on a single input.
		Accepts an image and a label, as well as a learning rate.
		This will forward propogate the image, determine the cost by
		using the correct label, and then add the gradientStep
		times the gradient to the weights.
		"""

		gradient = self.__dCost( trainImage, trainLabel )
		print( 'This training sample\'s gradient: {}'.format( gradient ) )
		# Add the gradient to each layer, times the gradient step:
		newLayers = []
		for layer in range( self.layers ):
			print( 'Layer Number: {}'.format( layer ) )
			newLayers.append( layer + gradient[layer] * gradientStep )

		self.layers = newLayers

	def forwardPropogate( self, inputVec ):
		layerOutputs = []
		layerActivations = []

		for layer in self.layers:
			if len( layerOutputs ) == 0:
				layerOutputs.append( numpy.dot( inputVec, layer ) )
			else:
				layerOutputs.append( numpy.dot( layerActivations[-1], layer ) )

			layerActivations.append( self.__relu( layerOutputs[-1] ) )

		return ( layerOutputs, layerActivations )

	def predict( self, inputVec ):
		"""
		Run a single prediction step.
		Basically, this forward propogates an input all the way through, then
		identifies the number that the output corresponds to.
		"""
		_, activations = self.forwardPropogate( inputVec )
		output = activations[-1]
		return numpy.dot( output == numpy.max( output ), numpy.array( [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] ) )

	def __cost( self, inputVec, labelVec ):
		return 0.5 * numpy.sum( numpy.power( self.forwardPropogate( inputVec ) - labelVec, 2 ) )

	def __dCost( self, inputVec, labelVec ):
		layerOutputs, layerActivations = self.forwardPropogate( inputVec )

		backpropError = []
		layerGradients = []

		for layer in range( len( self.layers ), 0, -1 ):
			print( 'Backpropogating layer {}'.format( layer ) )
			if layer == len( self.layers ):
				# Compute the backpropogation error:
				thisBackpropError = numpy.multiply( -( labelVec - layerActivations[-1] ), self.__dRelu( layerOutputs[-1] ) )
				backpropError.insert( 0, thisBackpropError )
				# print it all out:
				# print( 'Backpropogation Error: {}'.format( backpropError[0] ) )
				print( 'Backpropogation Error shape: {}'.format( backpropError[0].shape ) )
				print( 'Layer activations shape: {}'.format( layerActivations[layer - 1].shape ) )
				print( 'Layer activations transpose shape: {}'.format( layerActivations[layer - 1][:, numpy.newaxis].shape ) )

				# compute the layer gradient:
				thisLayerGradient = numpy.dot( layerActivations[layer - 1][:, numpy.newaxis], thisBackpropError )
				layerGradients.insert( 0, thisLayerGradient )
				print( 'Layer gradients: {}'.format( layerGradients ) )

			else:
				# print( numpy.dot( backpropError[-1], self.layers[layer].T ) )
				backpropError.insert( 0, numpy.dot( backpropError[0], self.layers[layer].T ) * self.__dRelu( layerOutputs[layer] ) )
				print( 'Backpropogation Error: {}'.format( backpropError ) )
				layerGradients.insert( 0, numpy.dot( inputVec.T, layerGradients[layer] ) )
				print( 'Layer gradients: {}'.format( layerGradients ) )

		return layerGradients

	# def gradientDescent( self, samples, labels ):
	# 	for item in zip( samples, labels ):
	# 		error = self.forwardPropogate( item[0] ) -
	#
	# 	pass

	# Activation functions:
	def __relu( self, x ):
		"""
		Just a rectified linear unit.
		"""
		return numpy.max( x, 0 )

	def __dRelu( self, x ):
		"""
		Returns 1.0 if x > 0, else returns 0.
		"""
		return 1 * ( x > 0 )
