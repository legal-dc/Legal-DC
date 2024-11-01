#ROUGE-L 阈值0.6
import nltk
import re
import jieba
import json
from rouge_chinese import Rouge
import itertools
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
        Rouge_result=[]
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
            BLEU_result.append(isAccuracy_BLEU)

            # Rouge
            threshold_Rouge = 0.6  # 阈值
            rouge = Rouge()
            hypothesis = ' '.join(rag_answer_cut_Rouge)
            reference = ' '.join(answer_cut_Rouge)
            scores = rouge.get_scores(hypothesis, reference)
            rouge_l = scores[0]['rouge-l']['f']
            isAccuracy_Rouge=int(rouge_l > threshold_Rouge)
            Rouge_result.append(isAccuracy_Rouge)

            #将结果保存
            instance['BLEU-2']=BLEU
            instance['isAccuracy_BLEU']=isAccuracy_BLEU
            instance['rouge-l']=rouge_l
            instance['isAccuracy_Rouge']=isAccuracy_Rouge
            self.detail.append(instance)

        result.append({'BLEU_Accuracy':sum(BLEU_result) / len(BLEU_result)})
        result.append({'Rouge_Accuracy':sum(Rouge_result) / len(Rouge_result)})
        return result


    def get_detail(self,outputpath):
        with open(outputpath,'w',encoding='utf-8') as file :
            json.dump(self.detail,file,ensure_ascii=False,indent=2)

def main():
    filepath='./data/rag_qad_wenxin_plugin_recall.json'
    outputpath='./data/rag_qad_wenxin_plugin_recall_accuracy.json'
    accuracy_cal=Accuracy(filepath)
    result=accuracy_cal.cal_accuracy()
    print("回答的准确度:"+str(result))
    accuracy_cal.get_detail(outputpath)

if __name__ == '__main__':
    main()