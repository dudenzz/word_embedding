# coding=utf-8
import numpy as np

from scipy.spatial.distance import cosine as _cosine_distance, euclidean
from numpy import argmax, argmin, max, min, sum, zeros, float32, array, mean, isnan, any, inf, power, ones
from iq.tools.mixins import LogMixin
from iq.core.model_wrapper import VectorModelWrap


def cosine(u, v):
    return 1 - _cosine_distance(u, v)


class IqSolverValueAnalyzer(LogMixin):
    def __init__(self, model, dimensions=500, out_file=None, processing_model='w2v', log_level=1, set_zeros=True,
                 weights=(0.5,) * 10, antonym_vector=None, synonym_vector=None):
        """
        Kontruktor solvera pytań IQ
        :param processing_model: typ przetwarzanych wektorów - 'w2v' (word2vec skip gram), 'ms' (multi-sense model),
                                 'rk' (relation knowledge powered model)
        :return: IqSolverValueAnalyzer
        """
        if processing_model not in ['w2v', 'ms', 'rk']:
            raise ValueError("Unknown processing model")

        self._processing_model = processing_model
        self.dimensions = dimensions
        self.model = model
        self._set_zeros = set_zeros

        self._log_level = log_level
        self._out_file = out_file

        self._weights = weights

        self._norm_alg = 'norm2'

        self._synonym_vector = synonym_vector
        self._antonym_vector = antonym_vector

    def _write_to_output(self, log_text):
        if self._log_level:
            print log_text

        if self._out_file is not None:
            self._out_file.write(log_text + "\n")

    def _normalize_value(self, value, values, question):
        question_index = ['an1', 'an2', 'cls', 'syn', 'ant'].index(question)

        if self._norm_alg == 'norm2':
            return value * self._weights[question_index * 2] +\
                   value / np.sqrt(np.sum(np.power(values, 2))) * self._weights[question_index * 2 + 1]
        elif self._norm_alg == 'sum':
            return value * self._weights[question_index * 2] +\
                   value / np.sum(values) * self._weights[question_index * 2 + 1]
        else:
            return value

    def _get_from_model(self, word):
        """
        Pobieranie danych z modelu, z obsługę zerowych wektorów dla nieznanych słów
        :param word: słowo do znalezienia w modelu
        :return: wektor dla słowa
        """
        if not self._set_zeros:
            return self.model.get_word_vector(word)
        else:
            try:
                return self.model.get_word_vector(word)
            except:
                if self._log_level >= 1:
                    print("Unknown word {}, setting zeros".format(word))
                return zeros(self.dimensions)

    def analyze_analogy_1(self, question, extracted_a, extracted_b, extracted_c, extracted_answers,
                          distance_fun=cosine, choice_fun=argmax, inner_choice_fun=max):
        self._write_to_output("ANALOGY-1")
        self._write_to_output("Question: {}".format(question['question']))
        self._write_to_output("Answers: {}".format(question['possible_answers']))

        a = self._get_from_model(extracted_a)
        b = self._get_from_model(extracted_b)
        c = self._get_from_model(extracted_c)
        answers_vectors = [self._get_from_model(word) for word in extracted_answers]

        # power - 0 norm1, 1 norm 2 (some)
        if distance_fun == cosine:
            answers_vectors = [answer_vector if sum(power(answer_vector, 2)) != 0 else ones(len(answer_vector))
                               for answer_vector in answers_vectors]

        question_vector = b - a + c
        distances = [distance_fun(question_vector, answer_vector) for answer_vector in answers_vectors]

        if isnan(distances).any():
            distances = [_distance if not isnan(_distance) else
                         inf if choice_fun == argmin else -inf for _distance in distances]

        self._write_to_output("Distances for answers, chosen max:")
        for i in range(len(answers_vectors)):
            self._write_to_output("{}: {}".format(extracted_answers[i], distances[i]))

        self._write_to_output("Closest to question vector: ")
        try:
            if self._log_level:
                self._write_to_output(str(self.model.most_similar(positive=[extracted_b, extracted_c],
                                                                  negative=[extracted_a])))
        except:
            self._write_to_output("Cannot print, unknown words")

        chosen = choice_fun(distances)

        self._write_to_output("Chosen answer: {}".format(extracted_answers[chosen]))
        self._write_to_output("Correct answer: {}".format(question['correct_answer']))

        return extracted_answers[chosen], self._normalize_value(inner_choice_fun(distances), distances, 'an1')

    def analyze_analogy_2(self, question, extracted_a, extracted_bs, extracted_c, extracted_ds,
                          distance_fun=cosine, choice_fun=argmax, inner_choice_fun=max):
        self._write_to_output("ANALOGY-2")
        self._write_to_output("Question: {}".format(question['question']))
        self._write_to_output("Answers: {}".format(question['possible_answers']))

        a = self._get_from_model(extracted_a)
        c = self._get_from_model(extracted_c)

        b_vectors = [self._get_from_model(word) for word in extracted_bs]
        d_vectors = [self._get_from_model(word) for word in extracted_ds]

        if distance_fun == cosine:
            b_vectors = [answer_vector if sum(power(answer_vector, 2)) != 0 else ones(len(answer_vector))
                         for answer_vector in b_vectors]
            d_vectors = [answer_vector if sum(power(answer_vector, 2)) != 0 else ones(len(answer_vector))
                         for answer_vector in d_vectors]

        question_vectors = [b - a + c for b in b_vectors]
        distances = []

        self._write_to_output("Distances for answers, chosen max:")
        for j in range(len(extracted_bs)):
            self._write_to_output("With {}".format(extracted_bs[j]))
            distances.append([distance_fun(question_vectors[j], d) for d in d_vectors])

            # o ja pierdolę jak nie optymalnie
            for sub_d in distances:
                if isnan(sub_d).any():
                    distances = [_distance if not isnan(_distance) else
                                 inf if choice_fun == argmin else -inf for _distance in sub_d]

            # for i in range(len(extracted_ds)):
            #     print "{}: {}".format(extracted_ds[i], distances[j][i])

            self._write_to_output("Closest to question vector: ")
            try:
                if self._log_level:
                    self._write_to_output(str(self.model.most_similar(positive=[extracted_bs[j], extracted_c],
                                                                      negative=[extracted_a])))
            except:
                self._write_to_output("Cannot print, unknown words")

        chosen = choice_fun(distances)

        self._write_to_output("Chosen answers: {} {}".format(extracted_bs[int(chosen / len(extracted_bs))],
                                                             extracted_ds[chosen % len(extracted_ds)]))
        self._write_to_output("Correct answers: {}".format(question['correct_answer']))

        return extracted_bs[int(chosen / len(extracted_bs))], extracted_ds[chosen % len(extracted_ds)],\
               self._normalize_value(inner_choice_fun(distances), distances, 'an2')

    def analyze_classification(self, question, extracted_answers, distance_fun=euclidean, choice_fun=argmax,
                               inner_choice_fun=max):
        self._write_to_output("CLASSIFICATION")
        self._write_to_output("Question: {}".format(question['question']))
        self._write_to_output("Answers: {}".format(question['possible_answers']))

        v = [self._get_from_model(word) for word in extracted_answers]
        question_vector = mean(v, 0)

        distances = [distance_fun(question_vector, answer_vector) for answer_vector in v]

        if isnan(distances).any():
            distances = [_distance if not isnan(_distance) else
                         inf if choice_fun == argmin else -inf for _distance in distances]

        self._write_to_output("Distances for answers, chosen max:")
        for i in range(len(extracted_answers)):
            self._write_to_output("{}: {}".format(extracted_answers[i], distances[i]))

        self._write_to_output("Closest to question vector (by cosine, not euclidean!): ")
        try:
            if self._log_level:
                self._write_to_output(str(self.model.most_similar(positive=extracted_answers)))
        except:
            self._write_to_output("Cannot print, unknown words")

        chosen = choice_fun(distances)

        self._write_to_output("Chosen answer: {}".format(extracted_answers[chosen]))
        self._write_to_output("Correct answer: {}".format(question['correct_answer']))

        return extracted_answers[chosen], self._normalize_value(inner_choice_fun(distances), distances, 'cls')

    def analyze_synonym(self, question, extracted_a, extracted_answers, distance_fun=euclidean, choice_fun=argmin,
                        inner_choice_fun=min):
        self._write_to_output("SYNONYM")
        self._write_to_output("Question: {}".format(question['question']))
        self._write_to_output("Answers: {}".format(question['possible_answers']))

        v = [self._get_from_model(word) for word in extracted_answers]
        question_vector = self._get_from_model(extracted_a)

        if self._synonym_vector is None:
            distances = [distance_fun(question_vector, answer_vector) for answer_vector in v]
        else:
            distances = [distance_fun(question_vector, answer_vector - self._synonym_vector) for answer_vector in v]

        if isnan(distances).any():
            distances = [_distance if not isnan(_distance) else
                         inf if choice_fun == argmin else -inf for _distance in distances]

        self._write_to_output("Distances for answers, chosen min:")
        for i in range(len(extracted_answers)):
            self._write_to_output("{}: {}".format(extracted_answers[i], distances[i]))

        self._write_to_output("Closest to question vector (by cosine, not euclidean!): ")
        try:
            if self._log_level:
                self._write_to_output(str(self.model.most_similar(positive=extracted_answers)))
        except:
            self._write_to_output("Cannot print, unknown words")

        chosen = choice_fun(distances)

        self._write_to_output("Chosen answer: {}".format(extracted_answers[chosen]))
        self._write_to_output("Correct answer: {}".format(question['correct_answer']))

        return extracted_answers[chosen], self._normalize_value(inner_choice_fun(distances), distances, 'syn')

    def analyze_antonym(self, question, extracted_a, extracted_answers, distance_fun=euclidean, choice_fun=argmin,
                        inner_choice_fun=min):
        self._write_to_output("ANTONYM")
        self._write_to_output("Question: {}".format(question['question']))
        self._write_to_output("Answers: {}".format(question['possible_answers']))

        v = [self._get_from_model(word) for word in extracted_answers]
        question_vector = self._get_from_model(extracted_a)

        if self._antonym_vector is None:
            distances = [distance_fun(question_vector, answer_vector) for answer_vector in v]
        else:
            distances = [distance_fun(question_vector, answer_vector - self._antonym_vector) for answer_vector in v]

        if isnan(distances).any():
            distances = [_distance if not isnan(_distance) else
                         inf if choice_fun == argmin else -inf for _distance in distances]

        self._write_to_output("Distances for answers, chosen min:")
        for i in range(len(extracted_answers)):
            self._write_to_output("{}: {}".format(extracted_answers[i], distances[i]))

        self._write_to_output("Closest to question vector (by cosine, not euclidean!): ")
        try:
            if self._log_level:
                self._write_to_output(str(self.model.most_similar(positive=extracted_answers)))
        except:
            self._write_to_output("Cannot print, unknown words")

        chosen = choice_fun(distances)

        self._write_to_output("Chosen answer: {}".format(extracted_answers[chosen]))
        self._write_to_output("Correct answer: {}".format(question['correct_answer']))

        return extracted_answers[chosen], self._normalize_value(inner_choice_fun(distances), distances, 'ant')


class IqSolverAnalyzer(LogMixin):
    def __init__(self, model, dimensions=500, out_file=None, processing_model='w2v', log_level=1, set_zeros=True,
                 weights=(0.5,) * 10, synonym_vector=None, antonym_vector=None):
        """
        Kontruktor solvera pytań IQ
        :param processing_model: typ przetwarzanych wektorów - 'w2v' (word2vec skip gram), 'ms' (multi-sense model),
                                 'rk' (relation knowledge powered model)
        :return: IqSolverAnalyzer
        """

        self.solver = IqSolverValueAnalyzer(model, dimensions, out_file, processing_model, log_level, set_zeros,
                                            weights, synonym_vector, antonym_vector)

    def analyze_analogy_1(self, question, extracted_a, extracted_b, extracted_c, extracted_answers,
                          distance_fun=cosine, choice_fun=argmax, inner_choice_fun=max):
        answer, _ = self.solver.analyze_analogy_1(question, extracted_a, extracted_b, extracted_c, extracted_answers,
                                                  distance_fun, choice_fun, inner_choice_fun)

        return answer

    def analyze_analogy_2(self, question, extracted_a, extracted_bs, extracted_c, extracted_ds,
                          distance_fun=cosine, choice_fun=argmax, inner_choice_fun=max):
        answer_1, answer_2, _ = self.solver.analyze_analogy_2(question, extracted_a, extracted_bs, extracted_c,
                                                              extracted_ds, distance_fun, choice_fun, inner_choice_fun)

        return answer_1, answer_2

    def analyze_classification(self, question, extracted_answers, distance_fun=euclidean, choice_fun=argmax,
                               inner_choice_fun=max):
        answer, _ = self.solver.analyze_classification(question, extracted_answers, distance_fun, choice_fun,
                                                       inner_choice_fun)

        return answer

    def analyze_synonym(self, question, extracted_a, extracted_answers, distance_fun=euclidean, choice_fun=argmin,
                        inner_choice_fun=min):
        answer, _ = self.solver.analyze_synonym(question, extracted_a, extracted_answers, distance_fun, choice_fun,
                                                inner_choice_fun)

        return answer

    def analyze_antonym(self, question, extracted_a, extracted_answers, distance_fun=euclidean, choice_fun=argmin,
                        inner_choice_fun=min):
        answer, _ = self.solver.analyze_antonym(question, extracted_a, extracted_answers, distance_fun, choice_fun,
                                                inner_choice_fun)

        return answer