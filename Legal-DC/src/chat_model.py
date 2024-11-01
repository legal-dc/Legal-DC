import requests
import json
from configparser import ConfigParser
from openai import OpenAI
import random
from http import HTTPStatus
import dashscope
class wenxin():
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key

        self.access_token = self.get_access_token()

    def get_access_token(self):
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        response = requests.post(url)
        return response.json().get("access_token")

    def chat_completion(self, query):
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={self.access_token}"

        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)

        return response.json()


class qwen():
    def chat_completion(self,query):
        messages = [
            {'role': 'user', 'content': query}]
        response = dashscope.Generation.call(
            'qwen1.5-110b-chat',
            messages=messages,
            # set the random seed, optional, default to 1234 if not set
            seed=random.randint(1, 10000),
            result_format='message',  # set the result to be "message" format.
        )
        # if response.status_code == HTTPStatus.OK:
        #     # print(response['output']['choices'][0]['message']['content'])
        # else:
        #     print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
        #         response.request_id, response.status_code,
        #         response.code, response.message
        #     ))
        return response['output']['choices'][0]['message']['content']

class qwen2():
    def chat_completion(self,query):
        client = OpenAI(
            base_url="http://10.122.231.38:8000/v1",
            api_key="token-abc123",
        )

        completion = client.chat.completions.create(
            model="Qwen2-7B-Instruct",
            messages=[
                {"role": "user", "content": query}
            ]
            )
        return completion.choices[0].message.content

class baichuan():
    
    def chat_completion(self,query):
        url = "https://api.baichuan-ai.com/v1/chat/completions"
        api_key = "sk-77ba704132e09cd2dcabeeb580801920"

        data = {
            "model": "Baichuan3-Turbo",
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "stream": False
        }

        json_data = json.dumps(data)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        }

        response = requests.post(url, data=json_data, headers=headers, timeout=60, stream=True)

        # if response.status_code == 200:
        #     print("请求成功！")
        #     print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        #     for line in response.iter_lines():
        #         if line:
        #             print(line.decode('utf-8'))
        # else:
        #     print("请求失败，状态码:", response.status_code)
        #     print("请求失败，body:", response.text)
        #     print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))

        return response.json()['choices'][0]['message']['content']
    


class OpenLLM():  
    def __init__(self, openai_api_key,openai_api_base):
         self.client = OpenAI(
             base_url=openai_api_base,
             api_key=openai_api_key,
             )
         
    def chat_completion(self, modelname, query):
        completion = self.client.chat.completions.create(
            model= modelname,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
                ],
            temperature = 0,
            max_tokens = 50
            )
        return completion.choices[0].message.content

def main(API_KEY, SECRET_KEY):
    chat_bot = wenxin(api_key=API_KEY, secret_key=SECRET_KEY)
    query = "介绍一下你自己"
    result = chat_bot.chat_completion(query)
    print(result)
    print("ans"+result['result'])

def openllm_main(openai_api_key,openai_api_base):
    modelname = '/root/Llama-2-7b-hf'
    chat_bot = OpenLLM(openai_api_key,openai_api_base)
    query = "介绍一下你自己"
    result = chat_bot.chat_completion(modelname,query)
    print("ans"+result['result'])

if __name__ == '__main__':
    config = ConfigParser()
    config.read('../token.config', encoding='UTF-8')
    api_key = config.get('tokens', 'API_KEY')
    secret_key = config.get('tokens', 'SECRET_KEY')
    main(api_key,secret_key)

    
