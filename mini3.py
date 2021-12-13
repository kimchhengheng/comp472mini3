import gensim
import pandas as pd
import numpy as np
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import random

model = ['word2vec-google-news-300','glove-wiki-gigaword-50','glove-twitter-50','glove-wiki-gigaword-100','glove-twitter-100','random']
dataframe = pd.read_csv('./synonyms.csv')
# write the data to file for word2vec model,part 1
def writeTofile(model,mode, dict):
    file_name ="./" +model
    with open(file_name, mode) as textfile:  
        for row in dict:
            textfile.writelines(row['question']+','+row['answer']+','+row['guessanswer']+','+row['labels']+"\n")
            if(row['index'] == 79):
                textfile.close()
file_name ="./analysis.csv" 
# write the data to file for mulitple model , part 2
def writeTofileAnalyze(modelname,writemode,size,correctnum,noguess,accuracy):
    with open(file_name, writemode ) as textfile:  
        textfile.writelines(modelname+','+size+','+correctnum+','+noguess+','+str(accuracy)+'\n')
            
        textfile.close()    

import gensim.downloader as api
for k in range(len(model)):
    if(model[k]!='random'):  
        wv = api.load(model[k])
    answerlist = []
    similarity = {}
    nonexisitinmodel ={}
    question =  None
    answer =  None
    questioninmodel= None
    labels= None
    guessanswer = None
    skip = None
    for i in range(len(dataframe)) :
        question =  dataframe.iloc[i, 0]
        answer =  dataframe.iloc[i, 1]
        questioninmodel= True
        labels= None
        guessanswer = None
        similarity = {}
        nonexisitinmodel ={}
        if(model[k] !='random'):    
            try:
                wv[question]
            except KeyError:
                questioninmodel = False
            if( questioninmodel):
                for j in range(6):
                    skip =False
                    if(j > 1):
                        guess = dataframe.iloc[i, j]
                        try:
                            wv[guess]
                        except KeyError:
                            nonexisitinmodel[guess] = True
                            skip=True
                        if(not skip):
                            similarity[guess]=wv.similarity(question, guess)
            
            if(not questioninmodel or len(nonexisitinmodel) == 4):
                labels = 'guess'
                guessanswer = dataframe.iloc[i, random.randint(2,5)]
            if(questioninmodel and len(nonexisitinmodel) != 4):
                similarity=  {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1] , reverse=True)}
                first_pair = list(similarity.items())[0]
                guessanswer = list(first_pair)[0]
        else:
            guessanswer =  dataframe.iloc[i, random.randint(2, 5)] 
        if(labels is None):
            labels = 'correct' if guessanswer == answer and answer is not None else 'wrong'
        answerlist.append({'index':i,'question':question,'answer':answer,'guessanswer':guessanswer,'labels':str(labels) })

     

    guesscount = 0 
    correctcount = 0
    for a in answerlist:
        if(a['labels'] == 'guess'):
            guesscount +=1
        elif(a['labels'] == 'correct'):
            correctcount +=1
    v = len(answerlist)-guesscount
    writemode = 'w' if k == 0 else 'a'
    if(model[k]!='random'):
        modelname= model[k]
        sizeofcorpus = str(len(wv.index_to_key))
    else:
        modelname ='random'
        sizeofcorpus = 0
    writeTofile(modelname+"-details.csv",'a',answerlist)
    writeTofileAnalyze(modelname,writemode,str(sizeofcorpus),str(correctcount),str(v),round(correctcount/v,16))

    

