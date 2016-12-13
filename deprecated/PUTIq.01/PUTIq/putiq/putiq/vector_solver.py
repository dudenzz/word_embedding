# coding=utf-8
from scipy.spatial.distance import cosine as _cosine_distance, euclidean
from numpy import argmax, argmin, max, min, sum, zeros, float32, array, inf, isnan, any, ones, power
from mixins import LogMixin


def cosine(u, v):
    return 1 - _cosine_distance(u, v)


class IqSolver(LogMixin):
    def __init__(self, dimensions=500, processing_model='w2v', log_level=0, synonym_vector=None, antonym_vector=None):
        """
        Kontruktor solvera pytań IQ
        :param processing_model: typ przetwarzanych wektorów - 'w2v' (word2vec skip gram), 'ms' (multi-sense model),
                                 'rk' (relation knowledge powered model)
        :return: IqSolver
        """
        if processing_model not in ['w2v', 'ms', 'rk']:
            raise ValueError("Unknown processing model")

        self._processing_model = processing_model
        self.dimensions = dimensions

        self._log_level = log_level

        self._synonym_vector = synonym_vector
        self._antonym_vector = antonym_vector

    def solve_analogy_1(self, a, b, c, answers_vectors, answers_names=None, distance_fun=cosine, choice_fun=argmax,
                        inner_choice_fun=max):
        """
        Rozwiązanie pytania typu analogy-1
        :param choice_fun: funkcja wyboru najlepszego wyniku (argmax, argmin)
        :param distance_fun: funkcja obliczania dystansu
        :param inner_choice_fun: funkcja redukcji wyników multisense (min, max)
        :param a: wektor A (dla w2v), lista wektorów znaczeń A (dla ms)
        :param b: wektor B (dla w2v), lista wektorów znaczeń B (dla ms)
        :param c: wektor C (dla w2v), lista wektorów znaczeń C (dla ms)
        :param answers_vectors: lista wektorów kolejnych odpowiedzi (dla w2v), lista list wektorów ms kolejnych
                                odpowiedzi (dla ms)
        :param answers_names: lista nazw słownych odpowiedzi lub None jeśli nie podane
        :return: nazwa słowna wybranej odpowiedzi jeśli podano answer_names, indeks odpowiedzi jeśli nie podano
        """
        if answers_names is not None and len(answers_names) != len(answers_vectors):
            raise ValueError("Vectors and names length not equal")

        if self._processing_model == 'w2v' or self._processing_model == 'rk':
            question_vector = b - a + c

            if distance_fun == cosine:
                answers_vectors = [answer_vector if sum(power(answer_vector, 2)) != 0 else ones(len(answer_vector))
                                  for answer_vector in answers_vectors]

            distances = [distance_fun(question_vector, answer_vector) for answer_vector in answers_vectors]

            if isnan(distances).any():
                distances = [_distance if not isnan(_distance) else
                             inf if choice_fun == argmin else -inf for _distance in distances]

            return choice_fun(distances) if answers_names is None else answers_names[choice_fun(distances)]
        elif self._processing_model == 'ms':
            # przyjmujemy że każde słowo może mieć wiele znaczeń, a zatem przygotowujemy się na listę wektorów zamiast
            # każdego wektora wejściowego

            question_vectors = [array(_b) - array(_a) + array(_c) for _b in b for _a in a for _c in c]
            distances = [
                inner_choice_fun([distance_fun(question_vector, answer_vector) for question_vector in question_vectors
                                  for answer_vector in answer_vectors_list]) for answer_vectors_list in answers_vectors]

            return choice_fun(distances) if answers_names is None else answers_names[choice_fun(distances)]
        else:
            raise NotImplementedError("Yyyyy?")

    def solve_analogy_2(self, a, c, b_candidate_vectors, d_candidate_vectors, b_candidate_names=None,
                        d_candidate_names=None, distance_fun=cosine, choice_fun=argmax, inner_choice_fun=max):
        """
        Rozwiązanie pytania typu analogy-2
        :param choice_fun: funkcja wyboru najlepszego wyniku (argmax, argmin)
        :param distance_fun: funkcja obliczania dystansu
        :param inner_choice_fun: funkcja redukcji wyników multisense (min, max)
        :param a: wektor A (dla w2v), lista wektorów znaczeń A (dla ms)
        :param c: wektor B (dla w2v), lista wektorów znaczeń B (dla ms)
        :param b_candidate_vectors: lista wektorów kolejnych odpowiedzi B (dla w2v), lista list wektorów kolejnych
                                    znaczeń odpowiedzi B (dla ms)
        :param d_candidate_vectors: lista wektorów kolejnych odpowiedzi D (dla w2v), lista list wektorów kolejnych
                                    znaczeń odpowiedzi D (dla ms)
        :param b_candidate_names: lista nazw słownych odpowiedzi B lub None jeśli nie podane
        :param d_candidate_names: lista nazw słownych odpowiedzi D lub None jeśli nie podane
        :return: para (nazwa B, nazwa D) jeśli podano nazwy odpowiedzi, para indeksów odpowiedzi jeśli nie podano
        """
        if (b_candidate_names is not None and len(b_candidate_names) != len(b_candidate_vectors)) or \
                (d_candidate_names is not None and len(d_candidate_names) != len(d_candidate_vectors)):
            raise ValueError("Vectors and names length not equal")

        if self._processing_model == 'w2v' or self._processing_model == 'rk':
            if distance_fun == cosine:
                b_candidate_vectors = [answer_vector if sum(power(answer_vector, 2)) != 0 else ones(len(answer_vector))
                                       for answer_vector in b_candidate_vectors]
                d_candidate_vectors = [answer_vector if sum(power(answer_vector, 2)) != 0 else ones(len(answer_vector))
                                       for answer_vector in d_candidate_vectors]

            question_vectors = [b - a + c for b in b_candidate_vectors]
            distances = [[distance_fun(question_vector, d) for d in d_candidate_vectors]
                         for question_vector in question_vectors]

            for sub_distance in distances:
                if isnan(sub_distance).any():
                    distances = [_distance if not isnan(_distance) else
                                 inf if choice_fun == argmin else -inf for _distance in sub_distance]

            answer_index = choice_fun(distances)

            # <3 python
            return (int(answer_index) / len(b_candidate_vectors), answer_index % len(
                    d_candidate_vectors)) if b_candidate_names is None or d_candidate_names is None else \
                (b_candidate_names[
                     int(answer_index) / len(b_candidate_vectors)], d_candidate_names[
                     answer_index % len(d_candidate_vectors)])
        elif self._processing_model == 'ms':
            # przyjmujemy że każde słowo może mieć wiele znaczeń, a zatem przygotowujemy się na listę wektorów zamiast
            # każdego wektora wejściowego

            known_senses = [array(_c) - array(_a) for _c in c for _a in a]

            question_vectors = [[array(b) + array(question_vector) for b in b_senses
                                 for question_vector in known_senses] for b_senses in b_candidate_vectors]

            distances = [
                [inner_choice_fun([distance_fun(d, q_v) for q_v in b_question_vectors for d in d_senses])
                 for d_senses in d_candidate_vectors] for b_question_vectors in question_vectors]

            answer_index = choice_fun(distances)

            return (int(answer_index) / len(b_candidate_vectors), answer_index % len(
                    d_candidate_vectors)) if b_candidate_names is None or d_candidate_names is None else \
                (b_candidate_names[
                     int(answer_index) / len(b_candidate_vectors)], d_candidate_names[
                     answer_index % len(d_candidate_vectors)])
        else:
            raise NotImplemented("Yyyyyy")

    def solve_classification(self, candidate_vectors, candidate_names=None, distance_fun=euclidean, choice_fun=argmax,
                             inner_choice_fun=max):
        """
        Rozwiązanie pytania typu klasyfikacja (wybierz niepasujące słowo)
        :param choice_fun: funkcja wyboru najlepszego wyniku (argmax, argmin)
        :param distance_fun: funkcja obliczania dystansu
        :param inner_choice_fun: funkcja redukcji wyników multisense (min, max)
        :param candidate_vectors: lista wektorów słów kandydatów (dla w2v), lista list sensów wektorów słów kandydatów
                                  (dla ms)
        :param candidate_names: lista słów kandydatów, None jeśli nie podano
        :return: wybrane niepasujące słowo, indeks jeśli nie podano listy słów
        """
        if candidate_names is not None and len(candidate_names) != len(candidate_vectors):
            raise ValueError("Vectors and names length not equal")

        if self._processing_model == 'w2v' or self._processing_model == 'rk':
            # po jednym wektorze dla każdego słowa
            mean_vector = sum(candidate_vectors, 0) / float(len(candidate_vectors))

            return choice_fun([distance_fun(mean_vector, candidate_vector) for candidate_vector in  # euclidan
                               candidate_vectors]) if candidate_names is None else candidate_names[
                choice_fun([distance_fun(mean_vector, candidate_vector) for candidate_vector in candidate_vectors])]
        elif self._processing_model == 'ms':
            mean_vectors = [zeros((self.dimensions,), dtype=float32)]
            n = len(candidate_vectors)

            for candidate_vectors_set in candidate_vectors:
                mean_vectors = [array(mean_vector) + array(candidate_vector) / n for mean_vector in mean_vectors for
                                candidate_vector in candidate_vectors_set]

            distances = [inner_choice_fun(
                    [distance_fun(candidate_vector, mean_vector) for mean_vector in mean_vectors for candidate_vector in
                     candidate_vectors_set]) for candidate_vectors_set in candidate_vectors]

            return choice_fun(distances) if candidate_names is None else candidate_names[choice_fun(distances)]
        else:
            raise NotImplementedError("Yyyyy")

    def solve_synonym(self, main_vector, candidate_vectors, candidate_names=None, distance_fun=euclidean,
                      choice_fun=argmin, inner_choice_fun=min):
        """
        Rozwiązanie pytania typu synonim
        :param choice_fun: funkcja wyboru najlepszego wyniku (argmax, argmin)
        :param distance_fun: funkcja obliczania dystansu
        :param inner_choice_fun: funkcja redukcji wyników multisense (min, max)
        :param main_vector: wektor słowa z pytania (dla w2v), lista wektorów znaczeń słowa pytania (dla ms)
        :param candidate_vectors: wektory słów odpowiedzi (dla w2v), lista list wektorów znaczeń słów odpowiedzi
                                  (dla ms)
        :param candidate_names: słowa z odpowiedzi, lub None jeśli nie podawane
        :return: indeks odpowiedzi jeśli nie podano candidate names, słowo odpowiedzi jeśli podano
        """
        if candidate_names is not None and len(candidate_vectors) != len(candidate_names):
            raise ValueError("Vectors and names length not equal")

        if self._processing_model == 'rk' and self._synonym_vector is not None:
            distances = [distance_fun(main_vector, candidate_vector - self._synonym_vector) for candidate_vector in
                         candidate_vectors]  # euclidean

            if isnan(distances).any():
                distances = [_distance if not isnan(_distance) else
                             inf if choice_fun == argmin else -inf for _distance in distances]

            return choice_fun(distances) if candidate_names is None else candidate_names[choice_fun(distances)]
        elif self._processing_model == 'w2v' or self._processing_model == 'rk':
            distances = [distance_fun(main_vector, candidate_vector) for candidate_vector in
                         candidate_vectors]  # euclidean

            if isnan(distances).any():
                distances = [_distance if not isnan(_distance) else
                             inf if choice_fun == argmin else -inf for _distance in distances]

            return choice_fun(distances) if candidate_names is None else candidate_names[choice_fun(distances)]
        elif self._processing_model == 'ms':
            distances = [inner_choice_fun(
                    [distance_fun(main_vector_meaning, candidate_vector_meaning) for main_vector_meaning in main_vector
                     for candidate_vector_meaning in candidate_vector]) for candidate_vector in candidate_vectors]

            return choice_fun(distances) if candidate_names is None else candidate_names[choice_fun(distances)]
        elif self._processing_model == 'rk':
            pass

    def solve_antonym(self, main_vector, candidate_vectors, candidate_names=None, distance_fun=euclidean,
                      choice_fun=argmin, inner_choice_fun=min):
        """
        Rozwiązanie pytania typu antonim
        :param choice_fun: funkcja wyboru najlepszego wyniku (argmax, argmin)
        :param distance_fun: funkcja obliczania dystansu
        :param inner_choice_fun: funkcja redukcji wyników multisense (min, max)
        :param main_vector: wektor słowa z pytania (dla w2v), lista wektorów znaczeń słowa pytania (dla ms)
        :param candidate_vectors: wektory słów odpowiedzi (dla w2v), lista list wektorów znaczeń słów odpowiedzi
                                  (dla ms)
        :param candidate_names: słowa z odpowiedzi, lub None jeśli nie podawane
        :return: indeks odpowiedzi jeśli nie podano candidate names, słowo odpowiedzi jeśli podano
        """
        if candidate_names is not None and len(candidate_vectors) != len(candidate_names):
            raise ValueError("Vectors and names length not equal")

        if self._processing_model == 'rk' and self._antonym_vector is not None:
            distances = [distance_fun(main_vector, candidate_vector - self._antonym_vector) for candidate_vector in
                         candidate_vectors]  # euclidean

            if isnan(distances).any():
                distances = [_distance if not isnan(_distance) else
                             inf if choice_fun == argmin else -inf for _distance in distances]

            return choice_fun(distances) if candidate_names is None else candidate_names[choice_fun(distances)]
        if self._processing_model == 'w2v' or self._processing_model == 'rk':
            distances = [distance_fun(main_vector, candidate_vector) for candidate_vector in candidate_vectors]

            if isnan(distances).any():
                distances = [_distance if not isnan(_distance) else
                             inf if choice_fun == argmin else -inf for _distance in distances]

            return choice_fun(distances) if candidate_names is None else candidate_names[choice_fun(distances)]
        elif self._processing_model == 'ms':
            distances = [inner_choice_fun(
                    [distance_fun(main_vector_meaning, candidate_vector_meaning) for main_vector_meaning in main_vector
                     for candidate_vector_meaning in candidate_vector]) for candidate_vector in candidate_vectors]

            return choice_fun(distances) if candidate_names is None else candidate_names[choice_fun(distances)]
        elif self._processing_model == 'rk':
            pass
