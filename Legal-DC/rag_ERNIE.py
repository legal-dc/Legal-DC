from src.chat_model import wenxin
import json
from tqdm import tqdm
from prompt import Prompt
API_KEY = "YOUR API KEY"
SECRET_KEY = "YOUR SECRET KEY"
chat_bot = wenxin(api_key=API_KEY, secret_key=SECRET_KEY)
rag_qad_ouput_file_path='./data/result/rag_qd_bce_BAAI_promt_test.json'
with open('./data/result/rag_qd_bce_BAAI_recall.json','r',encoding='utf-8') as f:
    data=json.load(f)
rag_qad=[]
i = -1
failed=[]
for item in tqdm(data[:5]):
    i+=1
    dic={}
    query=item['query']

    reference=item['retrieval']
    # for x in refence_docs:
    #     reference.append(x.page_content)
    knowledge_promt=Prompt()
    promt_query=knowledge_promt.knowledge(query,reference)
    rag_answer=chat_bot.chat_completion(promt_query)

    item['retrieval']=reference
    item['rag_answer']=rag_answer['result']
    #结果加入列表
    rag_qad.append(item)


#导出json文件
# print(failed)
with open(rag_qad_ouput_file_path,'w',encoding='utf-8')as json_file:
    json.dump(rag_qad,json_file,ensure_ascii=False,indent=2)
