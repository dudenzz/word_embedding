
# coding=utf-8
import gc


from putiq.model_wrapper import VectorModelWrap
from putiq.question_processor import QuestionProcessor


if __name__ == '__main__':
    same_questions = open("answered_questions.csv", "w")

    model_dir = "//home/kuba/vsm/models"

    model_file = "glove.840B.300d.ppv2.txt"
    glove = True  # odnosi się do formatu zapisu!
    binary = False
    dim = 300

    train_questions_dir = 'var_data/train'
    test_questions_dir = 'var_data/dest'

    model = VectorModelWrap(model_file, glove, binary, dim, model_dir)

    qp = QuestionProcessor(test_questions_dir, model, train_questions_dir, verbose=2,
                           multi_sense_model=None, solving_algorithm='w2v', zeroes_for_unknown=True)

    solve_result, classif_result, result_v, missing_questions, missing_answers = qp.test()
    print(model_file)
    print(solve_result, classif_result)

    same_questions.write(model_file + ";")
    for cq in result_v:
        same_questions.write("{};".format(cq))

    same_questions.write("\n")

    model = None
    gc.collect()
