import json
import nltk
import re
import jieba
import json
from rouge_chinese import Rouge
import itertools


class Accuracy():
    def __init__(self,data):
        self.data=data
        self.detail=[]

    def cal_accuracy(self):
        # 加载数据
        data=self.data

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

        # result.append({'BLEU_Accuracy':round(sum(BLEU_result_bool) / len(BLEU_result_bool),4)})
        result.append({'Rouge_Accuracy':round(sum(Rouge_result_bool) / len(Rouge_result_bool),4)})
        result.append({'BLEU_average':round(sum(BLEU_result) / len(BLEU_result),4)})
        result.append({'Rouge_average':round(sum(Rouge_result) / len(Rouge_result),4)})

        return result

    def get_detail(self,outputpath):
        with open(outputpath,'w',encoding='utf-8') as file:
            json.dump(self.detail,file,ensure_ascii=False,indent=2)

def main():
    type_list=[]
    file="rag_qad_QAnything_recall_baichuan_pro"
    

    #回答效果评估
    ##accuracey,BLEU,ROUGE
    filepath=f'./data/result/{file}.json'
    outputpath=f'./data/evaluate/{file}_class_eval.json'
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_Conceptual=[item for item in data if item.get('class')=='概念解释型']
    data_Generalization=[item for item in data if item.get('class')=='概括归纳型']
    data_Logic=[item for item in data if item.get('class')=='逻辑推理型']
    class_list=[data_Conceptual,data_Generalization,data_Logic]
    class_dic=['Conceptual','Generalization','Logic']

    def cal(name:str,data:list):
        accuracy_cal=Accuracy(data)
        result=accuracy_cal.cal_accuracy()
        print(f"{name}回答的准确度:"+str(result))
        return result

    with open(f'./data/evaluate/result/{file}_class_evalueate.text','w',encoding='utf-8') as ft:
        for i,data_class in enumerate(class_list):
            name=class_dic[i]
            result = cal(name,data_class)
            ft.write(f"{name}回答质量评估："+str(result))

    

if __name__=='__main__':
    main()