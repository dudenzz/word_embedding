__author__ = 'Asus'
__author__ = 'Asus'
from IClassifier import IClassifier
from utilities import load_stf
from glove import Glove
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from nltk.corpus import wordnet as wn


class GloveWordnetRetrofitClassifier(IClassifier):
    def __init__(self, filename, no_d):
        self.GloveInstance = Glove()
        self.GloveInstance = load_stf(filename, no_d)

    def answerQuestion(self,wordAskedFor,question,possibilities):
        qWs = self.converter(wordAskedFor)
        pWs = []
        for i,p in enumerate(possibilities):
            pWs.append([])
            for pS in self.converter(p):
                pWs[i].append(pS)

        pVs = []
        qVs = []
        maxSim = 0
        correct = -1
        comment = ''
        for i,pW in enumerate(pWs):
            pVs.append([])
            for p in pW:
                try:
                    pVs[i].append(self.GloveInstance.word_vectors[self.GloveInstance.dictionary[p]])
                except:
                    print p
        for q in qWs:
            try:
                qVs.append(self.GloveInstance.word_vectors[self.GloveInstance.dictionary[q]])
            except:
                print q
        for j,qV in enumerate(qVs):
            for i,pVn in enumerate(pVs):
                for k,pV in enumerate(pVn):
                    a = 1-cosine(qV,pV)
                    #a = 1/euclidean(qV,pV)
                    comment += '\n\t\t\tsim(' + qWs[j] + ',' + pWs[i][k] + ')=' +str(a)
                    if a>maxSim:
                        maxSim = a
                        correct = i
        print correct
        return (possibilities[correct],comment)
    def converter(self,word):
        wnFO = wn.abspath(wn._FILES[2]).open()
        ret = []
        for l in wnFO:
            if l.split('%')[0].strip() == word.strip():
                ret.append(l.split()[0])
        if len(ret) == 0:
            ret = [word+'%0:00:00::']
        return ret

