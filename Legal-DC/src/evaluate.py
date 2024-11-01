import json
import nltk
import re
import jieba
import json
from rouge_chinese import Rouge
import itertools

class RetrievalRecall:
    def __init__(self, filepath):
        self.filepath = filepath
        # 问题总数量
        self.query_nums = 0
        self.detail=[]

    def result(self):
        #加载数据
        with open(self.filepath,'r',encoding='utf-8') as f:
            rag_qad= json.load(f)
        self.query_nums=len(rag_qad)
        #命中数量
        n=0
        for item in rag_qad:
            search_result="".join(x for x in item['retrieval'])
            search_result=search_result.replace(" ","")
            search_result=search_result.replace("\\n","")
            standard_answer_list=item['document']
            # standard_answer_list=[]
            # standard_answer_list.append(item['document'])
            # print("#")
            # print(standard_answer_list)
            

            # 将标准答案的每一行都检查是否在检索结果中
            lines_in_search_result = all( line.replace(" ","") in search_result for line in standard_answer_list if line.strip())
            # 判断是否全部行都包含在检索结果中
            if lines_in_search_result:
                n+=1
            item['isRecall']=int(lines_in_search_result)
            self.detail.append(item)

        return n/self.query_nums

    def get_detail(self,outputpath):
        with open(outputpath,'w',encoding='utf-8') as file:
            json.dump(self.detail,file,ensure_ascii=False,indent=2)

class RetrievalMRR:
    def __init__(self, filepath):
        self.filepath = filepath
        # 问题总数量
        self.query_nums = 0

    def result(self):
        #加载数据
        with open(self.filepath,'r',encoding='utf-8') as f:
            rag_qad= json.load(f)
        self.query_nums=len(rag_qad)
        #总分
        score=0
        for item in rag_qad:
            record = [0] * 5
            # if item['isRecall']==1:
            standard_answer_list=item['document']
            # standard_answer_list =[]
            # standard_answer_list.append(item['document'])
            # print(standard_answer_list)
            num=len(standard_answer_list)
            for answer in standard_answer_list:
                answer_pro=answer.replace("\\n","")
                # print(answer)
                # print(answer_pro)
                for index,value in enumerate(item['retrieval']):
                    if answer_pro in value:
                        record[index]+=1
            sum=0
            for index2,value2 in enumerate(record):
                sum+=(1.0/(index2+1))*value2

            score+=sum/num

        return score/self.query_nums


class Accuracy():
    def __init__(self,filepath):
        self.filepath=filepath
        self.detail=[]

    def cal_accuracy(self):
        # 加载数据
        filepath=self.filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 字符串分词处理
        ##去符号处理
        def remove_punctuation_and_newline(text):
            # 去除标点符号和'\n'
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            cleaned_text = re.sub(r'\n', ' ', cleaned_text)
            return cleaned_text.replace(" ", "")

        n = 0
        BLEU_result = []
        BLEU_result_bool=[]
        Rouge_result=[]
        Rouge_result_bool=[]
        result=[]
        for instance in data:
            n += 1
            rag_answer = instance['rag_answer']
            answer = instance['answer']
            ##分词处理
            answer_cut = jieba.cut(remove_punctuation_and_newline(answer), cut_all=False)
            answer_cut_BLEU, answer_cut_Rouge = itertools.tee(answer_cut, 2)
            rag_answer_cut = jieba.cut(remove_punctuation_and_newline(rag_answer), cut_all=False)
            rag_answer_cut_BLEU, rag_answer_cut_Rouge = itertools.tee(rag_answer_cut, 2)

            # BLEU
            threshold_BLEU=0.4  #阈值
            answer_list=[value for value in answer_cut_BLEU]
            rag_answer_list=[value for value in rag_answer_cut_BLEU]
            BLEU = nltk.translate.bleu_score.sentence_bleu([answer_list],
            rag_answer_list,weights=(0.6,0.4))

            isAccuracy_BLEU=int(BLEU>threshold_BLEU)
            BLEU_result_bool.append(isAccuracy_BLEU)
            BLEU_result.append(BLEU)

            # Rouge
            threshold_Rouge = 0.6  # 阈值
            rouge = Rouge()
            hypothesis = ' '.join(rag_answer_cut_Rouge)
            reference = ' '.join(answer_cut_Rouge)
            scores = rouge.get_scores(hypothesis, reference)
            rouge_l = scores[0]['rouge-l']['f']
            isAccuracy_Rouge=int(rouge_l > threshold_Rouge)
            Rouge_result_bool.append(isAccuracy_Rouge)
            Rouge_result.append(rouge_l)

            #将结果保存
            instance['BLEU-2']=BLEU
            instance['isAccuracy_BLEU']=isAccuracy_BLEU
            instance['rouge-l']=rouge_l
            instance['isAccuracy_Rouge']=isAccuracy_Rouge
            self.detail.append(instance)
        
        # print(BLEU_result_bool)
        # print(BLEU_result)
        # print(Rouge_result_bool)
        # print(Rouge_result)

        result.append({'BLEU_Accuracy':sum(BLEU_result_bool) / len(BLEU_result_bool)})
        result.append({'Rouge_Accuracy':sum(Rouge_result_bool) / len(Rouge_result_bool)})
        result.append({'BLEU_average':sum(BLEU_result) / len(BLEU_result)})
        result.append({'Rouge_average':sum(Rouge_result) / len(Rouge_result)})

        return result

    def get_detail(self,outputpath):
        with open(outputpath,'w',encoding='utf-8') as file:
            json.dump(self.detail,file,ensure_ascii=False,indent=2)

class test1:
    pass

def main():
    file="rag_qad_5"
    #检索效果评估
    ##recall
    filepath_recall=f'../data/result/{file}.json'
    outputpath_recall=f'../data/evaluate/{file}_recall.json'
    recall = RetrievalRecall(filepath_recall)
    result_recall = recall.result()
    print("检索的召回率："+str(result_recall))
    recall.get_detail(outputpath_recall)

    ##mrr
    filepath_mrr=f'../data/evaluate/{file}_recall.json'
    mrr = RetrievalMRR(filepath_mrr)
    result_mrr = mrr.result()
    print("检索的MRR分数："+str(result_mrr))

     
    #回答效果评估
    ##accuracey,BLEU,ROUGE
    filepath_ans=f'../data/evaluate/{file}_recall.json'
    outputpath_ans=f'../data/evaluate/{file}_recall_ans.json'
    accuracy_cal=Accuracy(filepath_ans)
    result_ans=accuracy_cal.cal_accuracy()
    print("回答的准确度:"+str(result_ans))
    accuracy_cal.get_detail(outputpath_ans)

    with open(f'../data/evaluate/{file}_evalueate.text','w',encoding='utf-8') as ft:
        ft.write("检索评估\n")
        ft.write("检索召回分数："+str(result_recall)+"\n")
        ft.write("检索召MRR分数："+str(result_mrr)+"\n")
        ft.write("回答效果评估\n")
        ft.write("模型检索增强回答质量评估："+str(result_ans))

    

if __name__=='__main__':
    main()