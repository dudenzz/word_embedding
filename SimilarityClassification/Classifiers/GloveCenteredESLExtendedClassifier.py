___author__ = 'Asus'
from IClassifier import IClassifier
from Utils.utilities import load_stf
from glove import Glove
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import numpy as np

class GloveClassifier(IClassifier):
	def __init__(self):
		self.GloveInstace = None
		self.Centroids = None
	def answerQuestion(self,wordAskedFor,question,possibilities):
		qV = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[wordAskedFor]]
		pVs = []
		cqV = self.Centroids[wordAskedFor]
		cpVs = []
		maxSim = 1000
		correct = -1
		comment = ''
		for p in possibilities:
			pVs.append(self.GloveInstance.word_vectors[self.GloveInstance.dictionary[p]])
			cpVs.append(self.Centroids[p])
		for i,pV in enumerate(pVs):
			a = cosine(qV,pV) - np.power(cosine(pV,cpVs[i]),0.09)
			#a = 1/euclidean(qV,pV)
			nPtokens = question.split(' ')
			tokens = []
			for token in nPtokens:
				tokens.append(token.strip().strip('.'))
			wAi = -1
			for j,token in enumerate(tokens):
				if token == wordAskedFor:
					wAi = j
			for j in range(9):
				try:
					m1 = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[tokens[wAi-j-1]]]
					a += (1/np.power(j+1,2.0))*(cosine(pV,m1) - np.power(cosine(pV,cpVs[i]),0.09))
				except: nothing = 0
				try:
					d1 = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[tokens[wAi+j+1]]]
					a += (1/np.power(j+1,2.0))*(cosine(pV,d1) - np.power(cosine(pV,cpVs[i]),0.09))
				except: nothing = 0
			comment += '\n\t\t\tsim(' + wordAskedFor + ',' + possibilities[i] + ')=' +str(a)
			if a<maxSim:
				maxSim = a
				correct = i
		return (possibilities[correct],comment)

