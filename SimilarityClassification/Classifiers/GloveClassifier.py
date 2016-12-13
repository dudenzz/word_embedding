__author__ = 'Asus'
from IClassifier import IClassifier
from utilities import load_stf
from glove import Glove
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

class GloveClassifier(IClassifier):
    def __init__(self):
        self.GloveInstace = None
    def answerQuestion(self,wordAskedFor,question,possibilities):
        qV = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[wordAskedFor]]
        pVs = []
        maxSim = 0
        correct = -1
        comment = ''
        for p in possibilities:
            pVs.append(self.GloveInstance.word_vectors[self.GloveInstance.dictionary[p]])
        for i,pV in enumerate(pVs):
            a =1-cosine(qV,pV)
            #a = 1/euclidean(qV,pV)
	    comment += '\n\t\t\tsim(' + wordAskedFor + ',' + possibilities[i] + ')=' +str(a)
            if a>maxSim:
                maxSim = a
                correct = i

        return (possibilities[correct],comment)
