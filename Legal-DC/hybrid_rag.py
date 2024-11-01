from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from src.prompt import Prompt
import json
from src.chat_model import wenxin
from tqdm import tqdm
from collections import namedtuple
API_KEY = "YOUR API KEY"
SECRET_KEY = "YOUR SECRET KEY"
chat_bot = wenxin(api_key=API_KEY, secret_key=SECRET_KEY)

#加载数据
docs=[]
with open('./data/struct_document_539_final.json', 'r',encoding='utf-8') as file:
    docs=json.load(file)

# 初始化一个空的二元组列表
Document = namedtuple('PageTuple', ['page_content', 'metadata'])

#数据切片预处理
tuple_list = []
for doc in docs:
    title=doc['标题']
    metadata={'source': title}
    for index,content in doc['正文'].items():
        page_tuple = Document(page_content=content, metadata=metadata)
        tuple_list.append(page_tuple)

from BCEmbedding.tools.langchain import BCERerank
import os
import huggingface_hub
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


#embedding模型加载
model_name="maidalun1020/bce-embedding-base_v1"
print(f"开始加载embedding模型{model_name}")
embedding = HuggingFaceEmbeddings(model_name=model_name)
print(f"embedding模型{model_name}加载完成,开始加载rerank模型")

#rerank模型加载
reranker_args = {'model': 'maidalun1020/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda:0'}
reranker = BCERerank(**reranker_args)


#embedding,向量数据库存储
vectorstore_hf = Chroma.from_documents(documents=tuple_list, embedding=embedding , collection_name="huggingface_embed")

from src.bm_25 import retriever_bm25
from langchain_core.documents import Document
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for item in data:
        if item.page_content not in seen:
            unique_data.append(item)
            seen.add(item.page_content)
    return unique_data

def hybrid_retrieval(query:str):
    #bm25检索
    elasticsearch_url = "elasticsearch_url"
    index_name="law_documents"
    retriever = retriever_bm25(elasticsearch_url,index_name)
    retriever_bm25_result=retriever.retrieve(query)
    bm25_result_tuple_list=[]
    for item in retriever_bm25_result:
        metadata={'source': "bm25"}
        page_tuple = Document(page_content=item, metadata=metadata)
        bm25_result_tuple_list.append(page_tuple)

    #向量检索
    search_test=vectorstore_hf.similarity_search(query ,k = 10)
        
    #rerank测试
    documents_all=bm25_result_tuple_list+search_test
    documents_merge = remove_duplicates(documents_all)

    rerank_result=reranker.compress_documents(documents_merge,query)
    return rerank_result

#向量和BM25混合检索+rerank
#导入问题_答案_文档列表
with open('./data/pro_LawQA-2.json','r',encoding='utf-8') as qfile:
    q_a_d_list=json.load(qfile)

#获得问题_答案_文档_检索相关文档_RAG回答列表
rag_qad=[]

for item in tqdm(q_a_d_list):
    dic={}
    query=item['query']
    #获得检索文档以及RAG回答
    refence_docs = hybrid_retrieval(query)
    reference=[]
    for x in refence_docs:
        reference.append(x.page_content)
    knowledge_promt=Prompt()
    promt_query=knowledge_promt.knowledge(query,reference)
    try:
        rag_answer=chat_bot.chat_completion(promt_query)
    except:
        print("调用模型失败")
        print(item)
    item['retrieval']=reference
    item['rag_answer']=rag_answer['result']
    #结果加入列表
    rag_qad.append(item)


#导出json文件
rag_qad_ouput_file_path='./data/result/rag_qad_2_rerank.json'
with open(rag_qad_ouput_file_path,'w',encoding='utf-8')as json_file:
    json.dump(rag_qad,json_file,ensure_ascii=False,indent=2)