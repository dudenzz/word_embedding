try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np


class Glove(object):
    """
    Class for estimating GloVe word embeddings using the
    corpus coocurrence matrix.
    """

    def __init__(self, no_components=30, learning_rate=0.05,
                 alpha=0.75, max_count=100, max_loss=10.0,
                 random_state=None):
        """
        Parameters:
        - int no_components: number of latent dimensions
        - float learning_rate: learning rate for SGD estimation.
        - float alpha, float max_count: parameters for the
          weighting function (see the paper).
        - float max_loss: the maximum absolute value of calculated
                          gradient for any single co-occurrence pair.
                          Only try setting to a lower value if you
                          are experiencing problems with numerical
                          stability.
        - random_state: random statue used to intialize optimization
        """

        self.no_components = no_components
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.max_count = float(max_count)
        self.max_loss = max_loss

        self.word_vectors = None
        self.word_biases = None

        self.vectors_sum_gradients = None
        self.biases_sum_gradients = None

        self.dictionary = None
        self.inverse_dictionary = None

        self.random_state = random_state

    def add_dictionary(self, dictionary):
        """
        Supply a word-id dictionary to allow similarity queries.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of word vectors')

        self.dictionary = dictionary
        if hasattr(self.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = self.dictionary.iteritems()
        else:
            items_iterator = self.dictionary.items()

        self.inverse_dictionary = {v: k for k, v in items_iterator}

    @classmethod
    def load(cls, filename):
        """
        Load model from filename.
        """

        instance = Glove()

        with open(filename, 'rb') as savefile:
            instance.__dict__ = pickle.load(savefile)

        return instance

    def _similarity_query(self, word_vec, number):

        dst = (np.dot(self.word_vectors, word_vec)
               / np.linalg.norm(self.word_vectors, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)

        return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
                if x in self.inverse_dictionary]

    def most_similar(self, word, number=5):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            word_idx = self.dictionary[word]
        except KeyError:
            raise Exception('Word not in dictionary')

        return self._similarity_query(self.word_vectors[word_idx], number)[1:]

    def most_similar_multiple_words(self, positive=(), negative=(), number=5):
        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        query_vector = np.zeros(self.word_vectors.shape[1])

        for word in positive:
            try:
                query_vector += self.word_vectors[self.dictionary[word]]
            except KeyError:
                raise Exception('Word not in dictionary')

        for word in negative:
            try:
                query_vector -= self.word_vectors[self.dictionary[word]]
            except KeyError:
                raise Exception('Word not in dictionary')

        return self._similarity_query(query_vector, number)
