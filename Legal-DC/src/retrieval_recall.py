import json

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
            search_result =item['retrieval']
            standard_answer=item['document']
            # 将标准答案的每一行都检查是否在检索结果中
            lines_in_search_result = all( line.strip() in search_result for line in standard_answer.split('\n') if line.strip())

            # 判断是否全部行都包含在检索结果中
            if lines_in_search_result:
                n+=1
            item['isRecall']=int(lines_in_search_result)
            self.detail.append(item)

        return n/self.query_nums

    def get_detail(self,outputpath):
        with open(outputpath,'w',encoding='utf-8') as file:
            json.dump(self.detail,file,ensure_ascii=False,indent=2)

def main():
    filepath='./data/rag_qad.json'
    outputpath='./data/rag_qad_recall.json'
    recall = RetrievalRecall(filepath)
    result = recall.result()
    print("检索的召回率："+str(result))
    recall.get_detail(outputpath)

if __name__=='__main__':
    main()