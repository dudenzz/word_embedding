__author__ = 'Asus'

class QuestionBase:
    #possible question types:
    #Turney : "question [word].|ans1|ans2":cAns
    #TOEFL
    def __init__(self, filename, questionType):
        self.answers = []
        self.questionWords= []
        self.questions= []
        self.possibilities = []
        fObject = open(filename,'r')
        if questionType == 'Turney':
            for question in fObject:
                self.answers.append(question.split(':')[1])
                self.questionWords.append(question.split('[')[1].split(']')[0])
                self.questions.append(question.split('|')[0].strip('"'))
                possNonTrimmed = question.split('|')[1:5]
                possTrimmed = []
                for poss in possNonTrimmed:
                    possTrimmed.append(poss.split(':')[0].strip('"'))
                self.possibilities.append(possTrimmed)
        if questionType == "TOEFL":
            mode = 1 # 1 - questions; 2 - answers
            current = -1
            possibilities = []
            old = -1
            for question in fObject:
                if question.strip() == '':
                    continue
                if question[0].isdigit():
                    current = int(question.split('\t')[0].strip('.').strip())
                    if current<old:
                        mode = 2
                        self.possibilities.append(possibilities)
                    old = current
                if mode == 1:
                    if question[0].isdigit():
                        self.questions.append(question)
                        self.questionWords.append(question.split('\t')[1].strip())
                        if(current>1):
                            self.possibilities.append(possibilities)
                        possibilities = []
                    else:
                        possibilities.append(question.split('\t')[1].strip())
                if mode == 2:
                    ans = -1
                    char = question.split('\t')[3]
                    if char.strip() == 'a':
                        ans = 0
                    if char.strip() == 'b':
                        ans = 1
                    if char.strip() == 'c':
                        ans = 2
                    if char.strip() == 'd':
                        ans = 3

                    self.answers.append(self.possibilities[current-1][ans])




    def __str__(self):
        ret = ''
        for i in range(self.questions.__len__()):
            ret += 'Question: ' + str(self.questions[i]) + '\n'
            ret += '\tWord asked for: '+str(self.questionWords[i]) + '\n'
            ret += '\tPossibilities: ' + str(self.possibilities[i]) + '\n'
            ret += '\tCorrect Answer: ' + str(self.answers[i]) + '\n'
        return ret
    def classify(self,Classifier, oFile):
        prec = 0
        recall = 0
        total = 0
        for i in range(self.questions.__len__()):
            oFile.write('\n\nQuestion: ' + self.questions[i])
            oFile.write('\n\tQuestion word: ' + self.questionWords[i])
            oFile.write('\n\tPossibilities: ' + str(self.possibilities[i]))
            oFile.write('\n\tCorrect Answer: ' + self.answers[i])
            oFile.write('\tCalculation: ')
            total += 1
            try:
            	result = Classifier.answerQuestion(self.questionWords[i],self.questions[i],self.possibilities[i])
            	oFile.write('\n' + result[1])
            	cAns = result[0]
            	recall += 1
            	oFile.write('\n\tChosen answer: ' + cAns)
            	if self.answers[i].strip() == cAns.strip():
                	prec += 1
                	oFile.write('\n\t\tCORRECT!')
            	else:
                	oFile.write('\n\t\tWRONG!')
            except Exception as e:
		
                print(e.message)
                recall += 0

        oFile.write('\n\n\n\nTotal(flat): ' + str(total))
        oFile.write(' Precision(flat): ' + str(prec))
        oFile.write(' Recall(flat): ' + str(recall))
        oFile.write(' Precision: ' + str(prec/float(total)))
        oFile.write(' Recall: ' + str(recall/float(total)))
