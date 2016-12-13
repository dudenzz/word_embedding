__author__ = 'Jakub Dutkiewicz'
import numpy as np

class QuestionBase:

    def __init__(self, filename):
        self.word1 = []
        self.word2 = []
        self.sims = []
        iFile = open(filename)
        for line in iFile:
            self.word1.append(line.split(',')[0])
            self.word2.append(line.split(',')[1])
            self.sims.append(float(line.split(',')[2]))
    def __str__(self):
        ret = ''
        for i in range(self.word1.__len__()):
            ret += 'Word 1 ' + self.word1[i] + '\n'
            ret += 'Word 2 '+ self.word2[i] + '\n'
            ret += 'Similarity' + str(self.sims[i]) + '\n\n'
        return ret
    def evaluate(self):
        avgA = 0
        for w in self.sims:
            avgA += w
        avgA /= len(self.sims)
        stdDevA = 0
        for w in self.sims:
            stdDevA += np.power(w - avgA,2.0)
        stdDevA /= len(self.sims)
        stdDevA = np.sqrt(stdDevA)

        avgB = 0
        for w in self.simCalcs:
            avgB += w
        avgB /= len(self.simCalcs)
        stdDevB = 0
        for w in self.simCalcs:
            stdDevB += np.power(w - avgB,2.0)
        stdDevB /= len(self.simCalcs)
        stdDevB = np.sqrt(stdDevB)

        plotted = []
        for i in len(self.sims):
            plotted.append(self.sims[i]*self.simCalcs[i])

        EA = 0
        EB = 0
        EAB = 0
        for i in len(self.sims):
            EA += self.sims[i]
            EB += self.simCalcs[i]
            EAB += plotted[i]

        EA /= len(self.sims)
        EB /= len(self.sims)
        EAB /= len(self.sims)

        cov = EAB - (EA*EB)
        cor = cov / (stdDevA * stdDevB)
        return cor

    def classify(self,Classifier, oFile):
        self.simCalcs = []
        for i in range(self.word1.__len__()):
            similarity = Classifier.classify(self.word1[i],self.word2[i])
            self.simCalcs.append(similarity)
            oFile.write(self.word1 + " " + self.word2 + " " + str(self.sims[i]) + " " + str(self.simCalcs[i]) + "\n")
        oFile.write(str(self.evaluate()))
