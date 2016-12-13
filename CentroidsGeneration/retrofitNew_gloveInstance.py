__author__ = 'Asus'
import argparse
import gzip
import math
import numpy
import re
import sys
import time
import io
from glove import Glove
from copy import deepcopy
import numpy as np
isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()
def calculate_size(filename):

        start = time.clock()
        max = 0
        with io.open(filename, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                if(max%100000==0):
                    print(max,'stamp',time.clock() - start)
                max = max+1
        return max

def load_stf(filename, no_d):
        start = time.clock()
        dct = {}
        size = calculate_size(filename);
        vectors = np.ndarray(shape=(size,no_d), dtype=float)
        iter = 0
        with io.open(filename, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                tokens = line.strip().split(' ')
                if(iter%10000 == 0):
                    print(iter, size, 'stamp',time.clock() - start)
                word = tokens[0]
		try:
                	vectors[iter] = tokens[1:]
                except:
			print tokens
			if len(tokens)==300:
				word = ''
				vectors[iter] = tokens
			else:
			    print 'oops something went terribly wrong with this vector'
		dct[word] = i
                iter = iter+1
        # Infer word vectors dimensions.
        no_vectors = len(dct)
        print('stampnv',time.clock() - start)
        # Set up the model instance.
        instance = Glove()
        print('stampinst',time.clock() - start)
        instance.no_components = size
        print('stampnoc',time.clock() - start)
        instance.word_vectors = vectors
        print('stampwv',time.clock() - start)
        instance.word_biases = np.zeros(no_vectors)
        print('stampwb',time.clock() - start)
        instance.add_dictionary(dct)
        print('stampdict',time.clock() - start)
        return instance

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')
  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')
  for word, values in wordVectors.iteritems():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')
  outFile.close()

''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename, wordVecs):
  lexicon = {}
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
  return lexicon

''''''
def retrofit_new(glove_vsm,lexicon,numIters):
  newWordVecs = deepcopy(glove_vsm)
  wvWords = set(glove_vsm.dictionary.keys())
  loopVocab = wvWords.intersection(set(lexicon.keys()))
  for it in range(numIters):
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvWords)
      numN = len(wordNeighbours)
      if numN == 0:
        continue
      newVec = numN * glove_vsm.word_vectors[glove_vsm.dictionary[word]]
      for ppWord in wordNeighbours:
        newVec += newWordVecs.word_vectors[newWordVecs.dictionary[ppWord]]
      newWordVecs.word_vectors[newWordVecs.dictionary[word]] = newVec/(2*numN)
    return newWordVecs
''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue
      print word
      # the weight of the data estimate if the number of neighbours
      newVec = numNeighbours * wordVecs[word]
      # loop over neighbours and add to new vector (currently with weight 1)
      for ppWord in wordNeighbours:
        newVec += newWordVecs[ppWord]
      newWordVecs[word] = newVec/(2*numNeighbours)
  return newWordVecs

if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")
  parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
  parser.add_argument("-d", "--dims", type=int, default=10, help="Num dimensions")
  args = parser.parse_args()

  wordVecs = load_stf(args.input,args.dims)
  lexicon = read_lexicon(args.lexicon, wordVecs)
  numIter = int(args.numiter)
  outFileName = args.output
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  #print_word_vecs(retrofit(wordVecs, lexicon, numIter), outFileName)
  a = retrofit_new(wordVecs,lexicon,numIter)
  a.save(outFileName)
