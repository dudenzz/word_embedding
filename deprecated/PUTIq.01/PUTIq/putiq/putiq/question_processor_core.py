# coding=utf-8
from vector_solver import IqSolver
from numpy import array, zeros
from abc import abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
import os
import json


def l(s):
    s = s.lower()
    s = ''.join(s.split('-'))

    return s


class QuestionProcessorCore:
    def __init__(self, questions_directory, model, train_questions_directory,
                 labels=('Analogy-1', 'Analogy-2', 'Classification', 'Synonym', 'Antonym'), multi_sense_model=None,
                 solving_algorithm='w2v', verbose=2, zeroes_for_unknown=False, antonym_vector=None,
                 synonym_vector=None):
        """
        Konstruktor procesora przetwarzającego pytania
        :param questions_directory: lokalizacja pytań zapisanych w formacie json
        :param model_directory: lokalizacja modelu
        :param labels: etykiety typów pytań
        :param multi_sense_model: model multi sense (słownik)
        :param solving_algorithm: wybrany algorytm rozwiązywania pytań: w2v - sam word2vec, ms - multi sense,
                                  rk - relation knowledge - stosowany najwyższy dostępny
        :param classifier: model klasyfikatora wieloklasowego: ovo - one vs one, ovr - one vs rest, ovrh - one vs rest
                           hierarchiczny
        :param dimensions: długość wektorów zastosowanych w modelu
        :return: QuestionProcessor
        """
        self._questions_directory = questions_directory
        self._train_questions_directory = train_questions_directory
        self._verbose = verbose

        self.model = model
        self.multi_sense_model = multi_sense_model

        self._labels = labels

        if solving_algorithm not in ['w2v', 'ms', 'rk']:
            raise ValueError("Bad solving algorithm value")

        self._solving_algorithm = solving_algorithm
        self._zeroes_for_unknown = zeroes_for_unknown

        self._classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                                     ('svm', SVC(kernel='linear'))])

        train_set = self.read_questions_from_json(False)

        x_train = map(lambda x: x['question'], train_set)
        x_target = map(lambda x: x['type'], train_set)

        self._classifier.fit(x_train, x_target)

        try:
            self._solver = IqSolver(self.model.dims, self._solving_algorithm, synonym_vector=synonym_vector,
                                    antonym_vector=antonym_vector)
            self._vector_size = self.model.dims
        except:
            if self.model is not None:
                raise ValueError("dis is bad")

    def classify_question(self, question_text):
        """
        Klasyfikacja kategorii pytania w oparciu o jego treść
        :param question_text: tekst pytania
        :return: etykietę kategorii pytania
        """
        return self._classifier.predict([question_text])[0]

    def _get_from_model(self, word):
        """
        Pobieranie danych z modelu, z obsługę zerowych wektorów dla nieznanych słów
        :param word: słowo do znalezienia w modelu
        :return: wektor dla słowa
        """
        if not self._zeroes_for_unknown:
            return self.model.get_word_vector(word)
        else:
            try:
                return self.model.get_word_vector(word)
            except:
                if self._verbose >= 2:
                    print("Unknown word {}, setting zeros".format(word))
                return zeros(self._vector_size)

    def _word_in_model(self, word):
        return self.model.word_in_model(word)

    @abstractmethod
    def test(self, questions_list=None):
        pass

    def _augment_to_multisense(self, word):
        """
        Zwraca listę multi sense dla danego słowa (jeśli dostępne)
        :param word: słowo
        :return: lista wektorów znaczeń dla danego słowa
        """
        if not self.multi_sense_model or word not in self.multi_sense_model:
            return [self._get_from_model(word)]
        else:
            return [self._get_from_model(word)] + [array(s) for s in self.multi_sense_model[word]]

    @staticmethod
    def extract_analogy_1(question_with_data):
        """
        Ekstrahuje z pytania analogy-1 słowa a, b, c i listę kandydatów na d oraz poprawną odpowiedź
        :param question_with_data: słownik reprezentujący pytanie {question: str, possible_answers: str,
                                   correct_answer: str, type: str}
        :return: słowa a, b, c i listę kandydatów na d oraz poprawną odpowiedź
        """
        question = map(l, re.findall("[\w\-]+", question_with_data['question']))

        return question[0], question[3], question[5], map(l,
                                                          re.findall("[\w\-]+",
                                                                     question_with_data['possible_answers'])), l(
            question_with_data['correct_answer'])

    @staticmethod
    def extract_analogy_2(question_with_data):
        """
        Ekstrahuje z pytania analogy-1 słowa a, c i listy kandydatów na b, d oraz poprawną odpowiedź
        :param question_with_data: słownik reprezentujący pytanie {question: str, possible_answers: str,
                                   correct_answer: str, type: str}
        :return: słowa a, b, c i listę kandydatów na d oraz poprawną odpowiedź
        :param question_with_data:
        :return: a, lista kandydatów b, c, lista kandydatów d, poprawna odpowiedź
        """
        a, c = map(l, re.findall("[A-Z\-]{2,}", question_with_data['possible_answers']))
        b, d = map(lambda ls: map(l, re.findall("[\w\-]+", ls)),
                   re.findall("\([\w, \-]+\)", question_with_data['possible_answers']))

        return a, b, c, d, tuple(re.findall("[\w\-]+", l(question_with_data['correct_answer'])))

    @staticmethod
    def extract_synonym(question_with_data):
        """
        Ekstrahuje z pytania synonim słowo będące przedmiotem pytania, listę możliwych odpowiedzi, poprawną odpowiedź
        :param question_with_data: słownik reprezentujący pytanie {question: str, possible_answers: str,
                                   correct_answer: str, type: str}
        :return: słowo będące przedmiotem pytania, listę możliwych odpowiedzi, poprawną odpowiedź
        """
        a = l(re.findall("[\w\-]+", question_with_data['question'])[-1]) if 'capitals' not in question_with_data[
            'question'] else l(re.search("[A-Z\-]+", question_with_data['possible_answers']).group())
        b = map(l, re.findall("[\w\-]+", question_with_data['possible_answers'])) if 'capitals' not in \
                                                                                     question_with_data[
                                                                                         'question'] else map(l,
                                                                                                              re.findall(
                                                                                                                  "[a-z\-]+",
                                                                                                                  question_with_data[
                                                                                                                      'possible_answers']))
        return a, b, l(question_with_data['correct_answer'])

    @staticmethod
    def extract_antonym(question_with_data):
        """
        Ekstrahuje z pytania antonim słowo będące przedmiotem pytania, listę możliwych odpowiedzi, poprawną odpowiedź
        :param question_with_data: słownik reprezentujący pytanie {question: str, possible_answers: str,
                                   correct_answer: str, type: str}
        :return: słowo będące przedmiotem pytania, listę możliwych odpowiedzi, poprawną odpowiedź
        """
        a = l(re.findall("[\w\-]+", question_with_data['question'])[-1]) if 'capitals' not in question_with_data[
            'question'] else l(re.search("[A-Z\-]+", question_with_data['possible_answers']).group())
        b = map(l, re.findall("[\w\-]+", question_with_data['possible_answers'])) if 'capitals' not in \
                                                                                     question_with_data[
                                                                                         'question'] else map(l,
                                                                                                              re.findall(
                                                                                                                  "[a-z\-]+",
                                                                                                                  question_with_data[
                                                                                                                      'possible_answers']))
        return a, b, l(question_with_data['correct_answer'])

    @staticmethod
    def extract_classification(question_with_data):
        """
        Ekstrahuje z pytania classification listę możliwych odpowiedzi oraz poprawne sformatowaną poprawną odpowiedź
        :param question_with_data: słownik reprezentujący pytanie {question: str, possible_answers: str,
                                   correct_answer: str, type: str}
        :return: lista możliwych odpowiedzi, poprawna odpowiedź
        """
        return map(l, re.findall("[\w\-]+", question_with_data['possible_answers'])), l(question_with_data[
                                                                                            'correct_answer'])

    def read_questions_from_json(self, test=True):
        """
        Wczytuje listę pytań z zadanej lokalizacji w formacie json - {'question':str, 'possible_answers':str,
        'correct_answers':str}
        :param test: flaga test-train set
        :return: lista pytań w postaci słowników zgodnych z wejściowym formatem json
        """
        questions_list = []

        questions_directory = self._questions_directory if test else self._train_questions_directory

        for f in os.listdir(questions_directory):
            if re.search("\.json", f):
                json_file = open(os.path.join(questions_directory, f), 'r')
                question_list = json.loads(json_file.read())

                for question in question_list:
                    question['type'] = re.search("[A-Za-z0-9]+(\-[1-2])*", f).group(0)

                questions_list += question_list
                json_file.close()

        return questions_list
