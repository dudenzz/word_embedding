__author__ = 'Asus'
import sys
import getopt


from SimilarityRegression.QuestionHandling.QuestionBase import QuestionBase
from SimilarityRegression.Classifiers.GloveClassifier import GloveClassifier
from SimilarityClassification.Utils.utilities import load_stf
from SimilarityClassification.Utils.retrofitNew_gloveInstance import retrofit_new
from SimilarityClassification.Utils.retrofitNew_gloveInstance import read_lexicon
import numpy as np

help_message = '''
$ python questionAnswering.py -v <vectorsFile> -q <questionsFile> -d <dimensions> [-o outputFile] [-h]
-v or --vectors to specify path to the word vectors input file in Glove text format
-q or --questions to specify path to the questions input file in "Question...[questionWord]...Question.|answer1|answer2|answer3|answer4":correctAnswer" format
-o or --output to optionally set path to output word sense vectors file (<vectorsFile>.results is used by default)
-h or --help (this message is displayed)
-r or --retro 1 to run the retrofit postprocessing on the vector space model 0 to skip
'''

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def readCommandLineInput(argv):
    try:
        try:
            #specify the possible option switches
            opts, _ = getopt.getopt(argv[1:], "hv:q:o:d:t:r:l:", ["help", "vectors=","questions=", "output=", "dimensions=", "type=","retro=","lexicon="])
        except getopt.error, msg:
            raise Usage(msg)

        vectorsFile = None
        questionsFile = None
        outputFile = None
        type = "Turney"
        dims = 0
        setOutput = False
        retro = 1
        lexicon = None
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            elif option in ("-v", "--vectors"):
                vectorsFile = value
            elif option in ("-q", "--ontology"):
                questionsFile = value
            elif option in ("-o", "--output"):
                outputFile = value
                setOutput = True
            elif option in ("-d", "--dimensions"):
                dims = value
            elif option in ("-t", "--type"):
                type = value
	    elif option in ("-r","--retro"):
		retro = value
            elif option in ("-l","--lexicon"):
                lexicon = value
        if (vectorsFile==None) or (questionsFile==None) or (dims==None):
            raise Usage(help_message)
        else:
            if not setOutput:
                outputFile = vectorsFile + '.results'
            return (vectorsFile, questionsFile, dims, outputFile, type,retro, lexicon)
    except Usage, err:
        print str(err.msg)
        return 2


if __name__ == "__main__":
	commandParse = readCommandLineInput(sys.argv)
    #commandParse = ('glove.6B.50d.txt','toefl.qst',50,'util.results','TOEFL')
	if commandParse==2:
		sys.exit(2)
	qb = QuestionBase(commandParse[1])
	print(commandParse)
	instance = load_stf(commandParse[0],int(commandParse[2]))
	lexicon = read_lexicon(commandParse[6], instance)
	print "starting retrofit procedure"
	instance_r = retrofit_new(instance, lexicon, 10)
	print "retrofit done"
	classifier = GloveClassifier()
	classifier.GloveInstance = instance_r
	#classifier.Centroids = np.load('//mnt/raid0/kuba/vsm/models/centroids_dir/ppdb_centroids').item()
	oFile = open(commandParse[3],'w+')
	qb.classify(classifier,oFile)
	oFile.close()

