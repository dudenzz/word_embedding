# coding=utf-8
from question_processor_core import QuestionProcessorCore
from gensim.models import Word2Vec
from model_wrapper import VectorModelWrap
from models_params import model_params
from multiprocessing import Pool, Lock
from functools import partial

import os
import pickle
import traceback
import gc


def l(s):
    s = s.lower()
    s = ''.join(s.split('-'))

    return s


class QuestionProcessor(QuestionProcessorCore):
    def __init__(self, questions_directory, model, train_questions_directory,
                 labels=('Analogy-1', 'Analogy-2', 'Classification', 'Synonym', 'Antonym'), multi_sense_model=None,
                 solving_algorithm='w2v', verbose=2, zeroes_for_unknown=False, synonym_vector=None,
                 antonym_vector=None):
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
                                       multi_sense_model, solving_algorithm, verbose, zeroes_for_unknown,
                                       synonym_vector=synonym_vector, antonym_vector=antonym_vector)

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

        correctly_answered = []
        missing_question = []
        missing_word = []

        for question in questions_list:
            # TODO if question synonym / antonym curse
            # annotation = self.classify_question(question['question'])
            annotation = question['type']

            if annotation.lower() == question['type'].lower():
                svm_accuracy += 1.0
            elif self._verbose >= 1:
                print(annotation, question['type'], question['question'])
            svm_classified += 1.0

            # Analogy-1', 'Analogy-2', 'Classification', 'Synonym', 'Antonym'

            if annotation == 'Analogy-1':
                a1_q += 1
                a, b, c, d, correct_answer = self.extract_analogy_1(question)

                if self._word_in_model(a) and self._word_in_model(b) and self._word_in_model(c):
                    missing_question.append(0)
                else:
                    missing_question.append(1)

                if filter(lambda t: not t, [self._word_in_model(t_d) for t_d in d]):
                    missing_word.append(1)
                else:
                    missing_word.append(0)

                try:
                    if self._solving_algorithm == 'w2v' or self._solving_algorithm == 'rk':
                        answer = self._solver.solve_analogy_1(self._get_from_model(a), self._get_from_model(b),
                                                              self._get_from_model(c),
                                                              [self._get_from_model(_d) for _d in d], d)
                    else:
                        answer = self._solver.solve_analogy_1(self._augment_to_multisense(a),
                                                              self._augment_to_multisense(b),
                                                              self._augment_to_multisense(c),
                                                              [self._augment_to_multisense(_d) for _d in d], d)

                    if answer == correct_answer:
                        accuracy += 1.0
                        analogy_1_accuracy += 1.0
                        correctly_answered.append(1)
                    else:
                        correctly_answered.append(0)
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
                    print("WARNING: unknown word for model: {}".format(filter(lambda x: self._word_in_model(x),
                                                                              [a, b, c] + d)))
                    analogy_1_random += 1.0
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Analogy-2':
                a2_q += 1
                a, b, c, d, correct_answer = self.extract_analogy_2(question)

                if self._word_in_model(a) and self._word_in_model(c):
                    missing_question.append(0)
                else:
                    missing_question.append(1)

                if filter(lambda t: not t, [self._word_in_model(t_d) for t_d in d]) + \
                        filter(lambda t: not t, [self._word_in_model(t_b) for t_b in b]):
                    missing_word.append(1)
                else:
                    missing_word.append(0)

                try:
                    if self._solving_algorithm == 'w2v' or self._solving_algorithm == 'rk':
                        answer = self._solver.solve_analogy_2(self._get_from_model(a), self._get_from_model(c),
                                                              [self._get_from_model(_b) for _b in b],
                                                              [self._get_from_model(_d) for _d in d], b, d)
                    else:
                        answer = self._solver.solve_analogy_2(self._augment_to_multisense(a),
                                                              self._augment_to_multisense(c),
                                                              [self._augment_to_multisense(_b) for _b in b],
                                                              [self._augment_to_multisense(_d) for _d in d], b, d)

                    if answer == correct_answer:
                        accuracy += 1.0
                        analogy_2_accuracy += 1.0
                        correctly_answered.append(1)
                    else:
                        correctly_answered.append(0)
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
                    print("WARNING: unknown word for model: {}".format(filter(lambda x: self._word_in_model(x),
                                                                              [a] + b + [c] + d)))
                    analogy_2_random += 1.0
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Classification':
                c_q += 1
                a, correct_answer = self.extract_classification(question)

                if filter(lambda t: not t, [self._word_in_model(t_a) for t_a in a]):
                    missing_question.append(1)
                else:
                    missing_question.append(0)

                missing_word.append(0)

                try:
                    if self._solving_algorithm == 'w2v' or self._solving_algorithm == 'rk':
                        answer = self._solver.solve_classification([self._get_from_model(_a) for _a in a], a)
                    else:
                        answer = self._solver.solve_classification([self._augment_to_multisense(_a) for _a in a], a)

                    if answer == correct_answer:
                        accuracy += 1.0
                        classification_accuracy += 1.0
                        correctly_answered.append(1)
                    else:
                        correctly_answered.append(0)
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
                    print("WARNING: unknown word for model: {}".format(filter(lambda x: self._word_in_model(x), a)))

                    classification_random += 1  # a[random.randint(0, len(a) - 1)]
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Synonym':
                s_q += 1
                a, b, correct_answer = self.extract_synonym(question)

                if self._word_in_model(a):
                    missing_question.append(0)
                else:
                    missing_question.append(1)

                if filter(lambda t: not t, [self._word_in_model(t_b) for t_b in b]):
                    missing_word.append(1)
                else:
                    missing_word.append(0)

                try:
                    if self._solving_algorithm == 'w2v' or self._solving_algorithm == 'rk':
                        answer = self._solver.solve_synonym(self._get_from_model(a),
                                                            [self._get_from_model(_b) for _b in b], b)
                    elif self._solving_algorithm == 'ms':
                        answer = self._solver.solve_antonym(self._augment_to_multisense(a),
                                                            [self._augment_to_multisense(_b) for _b in b], b)
                    else:
                        print("WARNING: rk not implemented")
                        continue

                    if answer == correct_answer:
                        accuracy += 1.0
                        synonym_accuracy += 1.0
                        correctly_answered.append(1)
                    else:
                        correctly_answered.append(0)
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
                    print(
                    "WARNING: unknown word for model: {}".format(filter(lambda x: self._word_in_model(x), [a] + b)))

                    synonym_random += 1.0  # b[random.randint(0, len(b) - 1)]
                except Exception:
                    traceback.print_exc()

            elif annotation == 'Antonym':
                a_q += 1
                a, b, correct_answer = self.extract_synonym(question)

                if self._word_in_model(a):
                    missing_question.append(0)
                else:
                    missing_question.append(1)

                if filter(lambda t: not t, [self._word_in_model(t_b) for t_b in b]):
                    missing_word.append(1)
                else:
                    missing_word.append(0)

                try:
                    if self._solving_algorithm == 'w2v' or self._solving_algorithm == 'rk':
                        answer = self._solver.solve_antonym(self._get_from_model(a),
                                                            [self._get_from_model(_b) for _b in b], b)
                    elif self._solving_algorithm == 'ms':
                        answer = self._solver.solve_antonym(self._augment_to_multisense(a),
                                                            [self._augment_to_multisense(_b) for _b in b], b)
                    else:
                        print("WARNING: rk not implemented")
                        continue

                    if answer == correct_answer:
                        accuracy += 1.0
                        antonym_accuracy += 1.0
                        correctly_answered.append(1)
                    else:
                        correctly_answered.append(0)
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
                    print(
                    "WARNING: unknown word for model: {}".format(filter(lambda x: self._word_in_model(x), [a] + b)))

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

        return accuracy / answered, svm_accuracy / svm_classified, correctly_answered, missing_word, missing_question


def worker(in_data):
    query_mode = 'w2v'
    do_vectors = query_mode == 'rk'

    if isinstance(in_data, tuple):
        model_name, i = in_data

        # print("Reading {}".format(model_name))

        _model = VectorModelWrap(model_name, glove=False, binary=True, dims=300 - 1 - i)

        if do_vectors:
            vector_file = open("{}.avgv".format(model_name), "rb")

            antonym = pickle.load(vector_file)
            synonym = pickle.load(vector_file)

            vector_file.close()
        else:
            antonym, synonym = None, None
    else:
        # print("Reading {}".format(in_data))

        _model = VectorModelWrap(in_data, glove=model_params[in_data]['glove'], binary=model_params[in_data]['binary'],
                                 dims=model_params[in_data]['dims'])

        if do_vectors:
            vector_file = open("{}.avgv".format(in_data), "rb")

            antonym = pickle.load(vector_file)
            synonym = pickle.load(vector_file)

            vector_file.close()
        else:
            antonym, synonym = None, None

    # print("Querying...")

    qp = QuestionProcessor('var_data/dest', _model, 'var_data/train', verbose=1,
                           multi_sense_model=None, solving_algorithm=query_mode, zeroes_for_unknown=True,
                           antonym_vector=antonym, synonym_vector=synonym)

    solve_result, classif_result, result_v, missing_questions, missing_answers = qp.test()

    lock.acquire()

    print(in_data)
    print(solve_result, classif_result)

    lock.release()

    _model = None
    gc.collect()

    return [in_data] + result_v


def init_lock(l):
    global lock
    lock = l

if __name__ == '__main__':
    # https://docs.google.com/document/d/1bfBT2eXdGE5HktZbZvaLxvxi4GqfOr8OpVDs--i0mXQ/edit

    gc.enable()

    multi_proc = False

    same_questions = open("answered_questions_with_svd_and_norm_zeros.csv", "w")
    missing_questions_file = open("missing words in question.csv", "w")
    missing_answers_file = open("missing words in answers.csv", "w")

    model_dir = "j:\wiki"

    models = ["glove_large_1_by_no_cut_svd.bin", "glove_large_1_by_0.95_svd.bin", "glove_large_2_by_0.95_svd.bin",
              "glove_large_3_by_0.95_svd.bin", "glove_large_4_by_0.95_svd.bin", "glove_large_5_by_0.95_svd.bin",
              "glove_large_1_by_0.1_svd.bin", "glove_large_1_by_0.9_svd.bin", "glove_large_1_by_0.5_svd.bin",
              "glove_large_10_svd.bin", "glove_large_8_svd.bin", "wiki_random.model", "wiki.model",
              "wiki_china_params.model", "wiki_trimmed.model", "GoogleNews-vectors-negative300.bin", "en.model",
              "glove_small.bin", "glove_large.bin", "w2v.glovewiki.1872168.400.bin"]  #, "glove.42B.300d.txt",
              # "glove.840B.300d.txt"]

    # models = [mdl for mdl in model_params]

    models = ["relational_glove\\glove.840B.300d.out\\glove.840B.300d.out"]

    # models += ["glove_large_1_by_{}_svd.bin".format(i + 1) for i in range(60)] if not multi_proc else\
    #     [("glove_large_1_by_{}_svd.bin".format(i + 1), i) for i in range(60)]

    i = 0
    solving_algorithm = 'w2v'

    if not multi_proc:
        for i in range(len(models)):
            # model = os.path.join(model_dir, models[i])
            model = models[i]

            #_model = VectorModelWrap(model, glove=model_params[model]['glove'], binary=model_params[model]['binary'],
            #                         dims=model_params[model]['dims'])
            _model = VectorModelWrap(model, glove=False, binary=True, dims=300 - 1 - i)

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
            qp = QuestionProcessor('var_data/dest', _model, 'var_data/train', verbose=1,
                                   multi_sense_model=_multisense_model, solving_algorithm='w2v', zeroes_for_unknown=True)
            solve_result, classif_result, result_v, missing_questions, missing_answers = qp.test()
            print(models[i])
            print(solve_result, classif_result)
            '''
            _log.write("model:{} classifier:{} iq:{} classifier:{}\n".format(model, 'sklearn', result[0], result[1]))

            _log.close()
            '''

            same_questions.write(models[i] + ";")
            for cq in result_v:
                same_questions.write("{};".format(cq))

            same_questions.write("\n")

            '''
            missing_questions_file.write(models[i] + ";")
            for cq in missing_questions:
                missing_questions_file.write("{};".format(cq))

            missing_questions_file.write("\n")

            missing_answers_file.write(models[i] + ";")
            for cq in missing_answers:
                missing_answers_file.write("{};".format(cq))

            missing_answers_file.write("\n")
            '''

            _model = None
            gc.collect()
    else:
        out_lock = Lock()

        pool = Pool(processes=5, initializer=init_lock, initargs=(out_lock,))

        pool_result = pool.map(worker, models)
        pool.close()
        pool.join()

        for r in pool_result:
            for cq in r:
                same_questions.write("{};".format(cq))

            same_questions.write("\n")

    same_questions.close()
    missing_questions_file.close()
    missing_answers_file.close()
