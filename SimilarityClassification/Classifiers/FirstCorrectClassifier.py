__author__ = 'Asus'

from IClassifier import IClassifier

class FirstCorrectClassifier(IClassifier):
     def answerQuestion(self,wordAskedFor,question,possibilities):
         return (possibilities[0],'First Possible Classifier is random so don\'t expect proof')