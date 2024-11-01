import json
import re  
  
def keep_only_digits(s):  
    # 使用正则表达式替换所有非数字字符为空字符串  
    # \D 匹配任何非数字字符，[\D．] 匹配任何非数字字符或中文句号  
    return re.sub(r'[\D．]', '', s)  

with open('./data/result/rag_qad_bce_BAAI_baichuan_100-500.json','r',encoding='utf-8') as f:
    data=json.load(f)

def get_llm_refer(model_response:str):

    # 使用字符串分割来获取参考编号的部分
    # ref_str = model_response.replace('\n','').replace('，',',').split("参考检索结果:")[1]
    try:
        ref_str = model_response['rag_answer'].replace('：',':').replace('\n','').replace('，',',').split("参考检索结果:")[1]
    except:
        # print(model_response['rag_answer'].replace('：',':').replace('\n','').replace('，',',').split("参考检索结果:"))
        dic={
        'llm_retrieval_list':[],
        'is_llm_retrieval_accuracy':0
    }
        return dic

    # 使用字符串分割来获取各个编号
    ref_numbers = ref_str.split(",")

    # 将字符串转换为整数列表
    try:
        ref_numbers_int = [int(keep_only_digits(number)) for number in ref_numbers]
    except:
        dic={
        'llm_retrieval_list':[],
        'is_llm_retrieval_accuracy':0
    }
        return dic
    # 输出结果
    # print(ref_numbers_int)
    document=model_response['document']
    retrieval=model_response['retrieval']
    llm_retrieval=[]
    for i in ref_numbers_int:
        try:
            llm_retrieval.append(retrieval[i-1])
        except:
            print(i)
    # print(document)
    # print(llm_retrieval)
    # print("="*50)
    

    for i in range(len(document)):
        for j in range(len(llm_retrieval)):
            if(document[i]!=0 and llm_retrieval[j]!=0 and document[i] in llm_retrieval[j]):
                document[i]=0
                llm_retrieval[j]=0
    
    is_llm_retrieval_accuracy=0
    if(all(item ==0 for item in document) and all(item == 0 for item in llm_retrieval)):
        is_llm_retrieval_accuracy=1
    dic={
        'llm_retrieval_list':ref_numbers_int,
        'is_llm_retrieval_accuracy':is_llm_retrieval_accuracy
    }
    return dic

accuracy_num=0
recall_num=0

for item in data:
    # print(item['rag_answer'])
    if(item['isRecall']):
        recall_num+=1
        dic =get_llm_refer(item)
        item['llm_retrieval_list']=dic['llm_retrieval_list']
        item['is_llm_retrieval_accuracy'] = dic['is_llm_retrieval_accuracy']
        if(dic['is_llm_retrieval_accuracy']==1):
            accuracy_num+=1


llm_retrieval_accuracy=round(accuracy_num/recall_num,4)

# print(recall_num)
print(llm_retrieval_accuracy)
with open('./data/evaluate/llm_retrieval/rag_qad_bce_BAAI_baichuan_0-100_eval.json','w',encoding='utf-8') as fw:
    json.dump(data,fw,ensure_ascii=False)

