import requests
import json
from tqdm import tqdm
API_key = "YOUR API KEY"
API_secret = "YOUR SECRET KEY"
def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id="+API_key+"&client_secret="+API_secret

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/plugin/8yzmysg0e51uzkpq/?access_token=" + get_access_token()

#加载数据
with open('./data/pro_LawQA-5.json','r',encoding='utf-8') as f:
    q_a_d_list=json.load(f)
rag_qad_wenxin_plugin=[]

for q in tqdm(q_a_d_list):
    payload = json.dumps({
    "query": q['query'],
    "plugins": ["uuid-zhishiku"],
    "verbose": True
})
    headers = {
    'Content-Type': 'application/json'
}

    response = requests.request("POST", url, headers=headers, data=payload)

    result=response.json()
    reference=[]
    for i in result['meta_info']['response']['result']['responses']:
        reference.append(i['content'].replace("reference:", ""))
    q['retrieval']=reference
    q['rag_answer']=result['result']
    rag_qad_wenxin_plugin.append(q)

with open('./data/result/rag_qad_5_wenxin_plugin.json','w',encoding='utf-8') as file:
    json.dump(rag_qad_wenxin_plugin,file,ensure_ascii=False,indent=2)

