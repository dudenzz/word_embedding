__author__ = 'Asus'
from IClassifier import IClassifier
from utilities import load_stf
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
            
	    comment += '\n\t\t\tsim(' + wordAskedFor + ',' + possibilities[i] + ')=' +str(a)
            if a<maxSim:
                maxSim = a
                correct = i

        return (possibilities[correct],comment)
