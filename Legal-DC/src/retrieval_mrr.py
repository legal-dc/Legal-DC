import json

class RetrievalMRR:
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
        #总分
        score=0
        for item in rag_qad:
            record = [0] * 5
            if item['isRecall']==1:
                standard_answer = item['document']
                standard_answer_list = standard_answer.split('\\n')
                num=len(standard_answer_list)
                for answer in standard_answer_list:
                    for index,value in enumerate(item['retrieval']):
                        if answer in value:
                            record[index]+=1
                sum=0
                for index2,value2 in enumerate(record):
                   sum+=(1.0/(index2+1))*value2

                score+=sum/num



        return score/self.query_nums

    def get_detail(self,outputpath):
        with open(outputpath,'w',encoding='utf-8') as file:
            json.dump(self.detail,file,ensure_ascii=False,indent=2)

def main():
    filepath='./data/rag_qad_QAnything_recall_accuracy.json'
    outputpath='./data/rag_qad_QAnything_recall_accuracy_mrr.json'
    recall = RetrievalMRR(filepath)
    result = recall.result()
    print("检索的MRR："+str(result))
    recall.get_detail(outputpath)

if __name__=='__main__':
    main()