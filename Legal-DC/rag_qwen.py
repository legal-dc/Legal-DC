from src.chat_model import qwen
import json
from tqdm import tqdm
from src.prompt import Prompt
chat_bot=qwen()
with open('./data/result/rag_qd.json','r',encoding='utf-8') as f:
    data=json.load(f)
rag_qad=[]
for item in tqdm(data[:5]):
    dic={}
    query=item['query']

    reference=item['retrieval']
    # for x in refence_docs:
    #     reference.append(x.page_content)
    knowledge_promt=Prompt()
    promt_query=knowledge_promt.knowledge(query,reference)
    
    rag_answer=chat_bot.chat_completion(promt_query)
    # print("#4 chat finish")
    item['retrieval']=reference
    item['rag_answer']=rag_answer
    #结果加入列表
    rag_qad.append(item)


#导出json文件
rag_qad_ouput_file_path='./data/result/rag_qad_rerank_qwen.json'
with open(rag_qad_ouput_file_path,'w',encoding='utf-8')as json_file:
    json.dump(rag_qad,json_file,ensure_ascii=False,indent=2)