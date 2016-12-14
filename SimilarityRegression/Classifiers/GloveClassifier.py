from IClassifier import IClassifier
from glove import Glove
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean


class GloveClassifier(IClassifier):
	def __init__(self):
		self.GloveInstace = None
	def Similarity(self,word1,word2):
		try:
			qV = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[word1.lower()]]
        		pV = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[word2.lower()]]
        		return 1-cosine(qV,pV)
		except: return 0.5
