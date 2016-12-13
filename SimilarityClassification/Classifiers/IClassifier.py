__author__ = 'Asus'
from abc import ABCMeta, abstractmethod

class IClassifier:
    __metaclass__ = ABCMeta
    @classmethod
    def version(self): return "1.0"
    @abstractmethod
    def answerQuestion(self,wordAskedFor,question,possibilities): raise NotImplementedError