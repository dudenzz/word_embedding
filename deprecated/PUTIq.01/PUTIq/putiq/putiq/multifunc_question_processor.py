# coding=utf-8
from vector_analyzer import IqSolverAnalyzer
from question_processor_core import QuestionProcessorCore
from model_wrapper import VectorModelWrap
from models_params import model_params

import os
import pickle
import traceback
import gc


def l(s):
    s = s.lower()
    s = ''.join(s.split('-'))

    return s


class MultifuncQuestionProcessor(QuestionProcessorCore):
    def __init__(self, questions_directory, model, train_questions_directory, log_file=None,
                 labels=('Analogy-1', 'Analogy-2', 'Classification', 'Synonym', 'Antonym'), multi_sense_model=None,
                 solving_algorithm='w2v', verbose=2, zeroes_for_unknown=False):
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

        QuestionProcessorCore.__init__(self, questions_directory, model, train_questions_directory, labels,
                                       multi_sense_model, solving_algorithm, verbose, zeroes_for_unknown)
        try:
            self._solver = IqSolverAnalyzer(self.model, self.model.dims, log_file, self._solving_algorithm)
            self._vector_size = self.model.dims
        except:
            self._solver = IqSolverAnalyzer(self.model, self.model.dims, log_file, self._solving_algorithm)
            self._vector_size = self.model.dims

    def test(self, questions_list=None):
        """
        Testowanie mechanizmu rozwiązującego pytania - na zadanej, lub podstawowej liście
        :param questions_list: lista pytań w formacie {'question': str, 'possible_answers': str, 'correct_answer': str}
        :return: accuracy rozwiązania testu - float, accuracy klasyfikacji pytań - float, wektor poprawnie rozwiązanych
                 pytań
        """
        if questions_list is None:
            questions_list = self.read_questions_from_json()

        accuracy = 0.0
        answered = 0.0

        svm_accuracy = 0.0
        svm_classified = 0.0

        analogy_1_accuracy = 0.0
        analogy_1_answered = 0.0
        analogy_1_random = 0

        analogy_2_accuracy = 0.0
        analogy_2_answered = 0.0
        analogy_2_random = 0

        classification_accuracy = 0.0
        classification_answered = 0.0
        classification_random = 0

        synonym_accuracy = 0.0
        synonym_answered = 0.0
        synonym_random = 0

        antonym_accuracy = 0.0
        antonym_answered = 0.0
        antonym_random = 0

        a1_q, a2_q, c_q, s_q, a_q = 0, 0, 0, 0, 0

        for question in questions_list:
            # TODO if question synonym / antonym curse
            annotation = self.classify_question(question['question'])

            if annotation.lower() == question['type'].lower():
                svm_accuracy += 1.0
            elif self._verbose >= 1:
                print(annotation, question['type'], question['question'])
            svm_classified += 1.0

            # Analogy-1', 'Analogy-2', 'Classification', 'Synonym', 'Antonym'

            if annotation == 'Analogy-1':
                a1_q += 1
                a, b, c, d, correct_answer = self.extract_analogy_1(question)

                try:
                    answer = self._solver.analyze_analogy_1(question, a, b, c, d)

                    if answer == correct_answer:
                        accuracy += 1.0
                        analogy_1_accuracy += 1.0
                    answered += 1.0
                    analogy_1_answered += 1.0

                    if self._verbose == 2:
                        print("Extracted ANALOGY1 a:{} b:{} c:{} d:{}".format(a, b, c, d))
                        print("{} classified as Analogy-1:\n\r{}\n\r{}\n\r{} correct: {} {}".format(question['type'],
                                                                                                    question[
                                                                                                        'question'],
                                                                                                    question[
                                                                                                        'possible_answers'],
                                                                                                    answer,
                                                                                                    correct_answer,
                                                                                                    answer == correct_answer))
                except KeyError:
                    print("WARNING: unknown word for model: {}".format(filter(self._word_in_model,
                                                                              [a, b, c] + d)))
                    analogy_1_random += 1.0
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Analogy-2':
                a2_q += 1
                a, b, c, d, correct_answer = self.extract_analogy_2(question)

                try:
                    answer = self._solver.analyze_analogy_2(question, a, b, c, d)

                    if answer == correct_answer:
                        accuracy += 1.0
                        analogy_2_accuracy += 1.0
                    answered += 1.0
                    analogy_2_answered += 1.0

                    if self._verbose == 2:
                        print("Extracted ANALOGY2 a:{} b:{} c:{} d:{}".format(a, b, c, d))
                        print("{} classified as Analogy_2:\n\r{}\n\r{}\n\r{} correct: {} {}".format(question['type'],
                                                                                                    question[
                                                                                                        'question'],
                                                                                                    question[
                                                                                                        'possible_answers'],
                                                                                                    answer,
                                                                                                    correct_answer,
                                                                                                    answer == correct_answer))
                except KeyError:
                    print("WARNING: unknown word for model: {}".format(filter(self._word_in_model,
                                                                              [a] + b + [c] + d)))
                    analogy_2_random += 1.0
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Classification':
                c_q += 1
                a, correct_answer = self.extract_classification(question)

                try:
                    answer = self._solver.analyze_classification(question, a)

                    if answer == correct_answer:
                        accuracy += 1.0
                        classification_accuracy += 1.0
                    answered += 1.0
                    classification_answered += 1.0

                    if self._verbose == 2:
                        print("Extracted Classification a:{}".format(a))
                        print("{} classified as Classification:\n\r{}\n\r{}\n\r{} correct: {} {}".format(
                            question['type'],
                            question[
                                'question'],
                            question[
                                'possible_answers'],
                            answer,
                            correct_answer,
                            answer == correct_answer))
                except KeyError:
                    print("WARNING: unknown word for model: {}".format(filter(self._word_in_model, a)))

                    classification_random += 1  # a[random.randint(0, len(a) - 1)]
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Synonym':
                s_q += 1
                a, b, correct_answer = self.extract_synonym(question)

                try:
                    answer = self._solver.analyze_synonym(question, a, b)

                    if answer == correct_answer:
                        accuracy += 1.0
                        synonym_accuracy += 1.0
                    answered += 1.0
                    synonym_answered += 1.0

                    if self._verbose == 2:
                        print("Extracted Synonym a:{} b:{}".format(a, b))
                        print("{} classified as Synonym:\n\r{}\n\r{}\n\r{} correct {} {}".format(question['type'],
                                                                                                 question['question'],
                                                                                                 question[
                                                                                                     'possible_answers'],
                                                                                                 answer,
                                                                                                 correct_answer,
                                                                                                 answer == correct_answer))
                except KeyError:
                    print("WARNING: unknown word for model: {}".format(filter(self._word_in_model, [a] + b)))

                    synonym_random += 1.0  # b[random.randint(0, len(b) - 1)]
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Antonym':
                a_q += 1
                a, b, correct_answer = self.extract_synonym(question)

                try:
                    answer = self._solver.analyze_antonym(question, a, b)

                    if answer == correct_answer:
                        accuracy += 1.0
                        antonym_accuracy += 1.0
                    answered += 1.0
                    antonym_answered += 1.0

                    if self._verbose == 2:
                        print("Extracted Antonym a:{} b:{}".format(a, b))
                        print("{} classified as Antonym:\n\r{}\n\r{}\n\r{} correct {} {}".format(question['type'],
                                                                                                 question['question'],
                                                                                                 question[
                                                                                                     'possible_answers'],
                                                                                                 answer,
                                                                                                 correct_answer,
                                                                                                 answer == correct_answer))
                except KeyError:
                    print("WARNING: unknown word for model: {}".format(filter(self._word_in_model, [a] + b)))

                    antonym_random += 1.0  # b[random.randint(0, len(b) - 1)]
                except Exception:
                    traceback.print_exc()

            else:
                print("WARNING: weird classifier stuff")

        if self._verbose:
            print("ACCURACY")
            print("ANALOGY-1: {} not answered: {} / {} correct {} answered {}".format(
                analogy_1_accuracy / analogy_1_answered if analogy_1_answered > 0 else 0,
                analogy_1_random, a1_q, analogy_1_accuracy, analogy_1_answered))
            print("ANALOGY-2: {} not answered: {} / {} correct {} answered {}".format(
                analogy_2_accuracy / analogy_2_answered if analogy_2_answered > 0 else 0,
                analogy_2_random, a2_q, analogy_2_accuracy, analogy_2_answered))
            print("CLASSIFICATION: {} not answered: {} / {} correct {} answered {}".format(
                classification_accuracy / classification_answered if classification_answered > 0 else 0,
                classification_random, c_q, classification_accuracy, classification_answered))
            print("SYNONYM: {} not answered: {} / {} correct {} answered {}".format(
                synonym_accuracy / synonym_answered if synonym_answered > 0 else 0, synonym_random, s_q,
                synonym_accuracy, synonym_answered))
            print("ANTONYM: {} not answered: {} / {} correct {} answered {}".format(
                antonym_accuracy / antonym_answered if antonym_answered > 0 else 0, antonym_random, a_q,
                antonym_accuracy, antonym_answered))
            print("Overall accuracy: {} not answered: {} / {} correct {} answered {}".format(
                accuracy / answered if answered > 0 else 0,
                analogy_1_random +
                analogy_2_random + classification_random +
                synonym_random + antonym_random,
                a1_q + a2_q + s_q + a_q + c_q,
                analogy_2_accuracy + analogy_1_accuracy + classification_accuracy + synonym_accuracy + antonym_accuracy,
                analogy_1_answered + analogy_2_answered + classification_answered + synonym_answered + antonym_answered))
            print("Classifier accuracy: {}".format(svm_accuracy / svm_classified))

        return accuracy / answered, svm_accuracy / svm_classified


if __name__ == '__main__':
    # https://docs.google.com/document/d/1bfBT2eXdGE5HktZbZvaLxvxi4GqfOr8OpVDs--i0mXQ/edit

    gc.enable()

    model_dir = "j:\wiki"

    models = ["wiki_random.model", "wiki.model", "wiki_china_params.model",
              "wiki_trimmed.model", "GoogleNews-vectors-negative300.bin", "en.model",
              "glove_small.bin", "glove_large.bin",
              "w2v.glovewiki.1872168.400.bin", "glove.42B.300d.txt", "glove.840B.300d.txt"]

    i = 0
    solving_algorithm = 'w2v'

    for i in [2, 7, 8, 9, 10]:
        model = models[i]
        log_file = open("{}_log.txt".format(models[i]), "w")

        _model = VectorModelWrap(model, glove=model_params[model]['glove'], binary=model_params[model]['binary'],
                                 dims=model_params[model]['dims'])

        if solving_algorithm == 'ms':
            ms_file = open("test/multisense_vectors.pckl")
            _multisense_model = pickle.load(ms_file)
            ms_file.close()
        else:
            _multisense_model = None

        # chwilowo bez logowania
        '''
        _log = open("questions_log_{}.txt".format(i), "w")
        '''
        qp = MultifuncQuestionProcessor('var_data/dest', _model, 'var_data/train', log_file=log_file, classifier=None, verbose=1,
                                        multi_sense_model=_multisense_model, solving_algorithm='w2v', zeroes_for_unknown=True)
        solve_result, classif_result = qp.test()
        print(models[i])
        print(solve_result, classif_result)

        _model = None
        log_file.close()
        gc.collect()
