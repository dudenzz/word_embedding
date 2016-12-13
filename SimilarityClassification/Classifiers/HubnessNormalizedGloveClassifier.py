__author__ = 'Asus'
from IClassifier import IClassifier
from utilities import load_stf
from glove import Glove
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import numpy as np

class GloveClassifier(IClassifier):
    def __init__(self,k):
        self.GloveInstance = None
        self.k = k+1
    def average_knn_distance(self,word):
	total = 0
        for w in self.GloveInstance.most_similar(word,self.k):
		total += w[1]
	return total/self.k
    def calculate_centroid(self):
        total = np.zeros(300)
        for w in self.GloveInstance.word_vectors:
            total += w
        self.centroid = total/len(self.GloveInstance.word_vectors)
    def answerQuestion(self,wordAskedFor,question,possibilities):
        qV = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[wordAskedFor]]-self.centroid
	pVs = []
        maxSim = 0
        correct = -1
        comment = ''
        distances = []
        for p in possibilities:
            print 'working'
            pVs.append((0,self.GloveInstance.word_vectors[self.GloveInstance.dictionary[p]]-self.centroid))
        for i,pV in enumerate(pVs):
            a =1-cosine(qV,pV[1])
	    comment += '\n\t\t\tsim(' + wordAskedFor + ',' + possibilities[i] + ')=' +str(a)
            if a>maxSim:
                maxSim = a
                correct = i

        return (possibilities[correct],comment)
