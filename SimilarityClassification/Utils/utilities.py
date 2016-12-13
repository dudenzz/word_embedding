__author__ = 'Asus'
import time
from glove import Glove
import numpy as np
import io

def calculate_size(filename):

        start = time.clock()
        max = 0
        with io.open(filename, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                if line.strip() == "":
		     continue;
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
		if (line.strip() == ""):
			continue;
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
				raise
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
