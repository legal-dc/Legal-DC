import json
import nltk
import re
import jieba
import json
# from rouge_chinese import Rouge
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

        return round(n/self.query_nums,4)

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

        return round(score/self.query_nums,4)



def main():
    file="rag_qd_bce_BAAI"
    #检索效果评估
    ##recall
    filepath_recall=f'./data/result/{file}.json'
    outputpath_recall=f'./data/evaluate/rerank/{file}_recall.json'
    recall = RetrievalRecall(filepath_recall)
    result_recall = recall.result()
    print("检索的召回率："+str(result_recall))
    recall.get_detail(outputpath_recall)

    ##mrr
    filepath_mrr=f'./data/evaluate/rerank/{file}_recall.json'
    mrr = RetrievalMRR(filepath_mrr)
    result_mrr = mrr.result()
    print("检索的MRR分数："+str(result_mrr))
    print(file)


    with open(f'./data/evaluate//rerank/result/{file}_evalueate.text','w',encoding='utf-8') as ft:
        ft.write("检索评估\n")
        ft.write("检索召回分数："+str(result_recall)+"\n")
        ft.write("检索召MRR分数："+str(result_mrr)+"\n")
 

    

if __name__=='__main__':
    main()