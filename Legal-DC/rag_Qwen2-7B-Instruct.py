from src.chat_model import qwen2
import json
from tqdm import tqdm
from src.prompt import Prompt
chat_bot=qwen2()
with open('./data/rag_qad_QAnything_recall.json','r',encoding='utf-8') as f:
    data=json.load(f)
rag_qad=[]
for item in tqdm(data):
    dic={}
    query=item['query']
    reference=['','','','','']
    reference_p=item['retrieval']
    for i,ref in enumerate(reference_p):
        reference[i]=reference_p[i]
    knowledge_promt=Prompt()
    promt_query=knowledge_promt.knowledge(query,reference)
    
    rag_answer=chat_bot.chat_completion(promt_query)
    item['retrieval']=reference
    item['rag_answer']=rag_answer
    #结果加入列表
    rag_qad.append(item)


#导出json文件
rag_qad_ouput_file_path='./data/result/rag_qad_QAnything_qwen2-7b.json'
with open(rag_qad_ouput_file_path,'w',encoding='utf-8')as json_file:
    json.dump(rag_qad,json_file,ensure_ascii=False,indent=2)