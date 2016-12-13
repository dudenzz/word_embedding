from SimilarityRegression.Classifiers.IClassifier import IClassifier
from SimilarityClassification.Utils.utilities import load_stf
from glove import Glove
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean


class GloveClassifier(IClassifier):
    def __init__(self):
        self.GloveInstace = None
    def Similarity(self,word1,word2):
        qV = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[word1]]
        pV = self.GloveInstance.word_vectors[self.GloveInstance.dictionary[word2]]
        return cosine(qV,pV)
