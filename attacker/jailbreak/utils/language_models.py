import openai
import time
import torch
import gc
from typing import Dict, List
import ssl
import hmac
import json
import base64
import hashlib
import datetime
import websocket  # 使用websocket_client
from time import mktime
import _thread as thread
from datetime import datetime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
import zhipuai


class LanguageModel:
    def __init__(self, model_name,api_key):
        self.model_name = model_name
        self.api_key=api_key

    def batched_generate(self, **kwargs):  # prompts_list: List, max_n_tokens: int, temperature: float):
        """
        按照批次生成信息
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        super().__init__(model_name)
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(
            self,
            full_prompts_list,
            max_n_tokens: int,
            temperature: float,
            top_p: float = 1.0
    ):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913,
            9092,
            16675
        ])


class Zhipu(LanguageModel):
    API_RETRY_SLEEP = 0
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    # TODO:别忘了删除key
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        api_key=self.api_key
        # print(type(api_key))
        self.client = zhipuai.ZhipuAI(api_key=api_key)

    def generate(
            self,
            conv: List[Dict],
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
        Returns:
            str: 生成的回复
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model='glm-4',
                    messages=conv,
                    stream=True,
                )
                output = ""
                for chunk in response:
                    output += chunk.choices[0].delta.content
                break
            except zhipuai.APIReachLimitError as e:
                print("ZhipuAI:", type(e), e)
        return output

    def batched_generate(
            self,
            conv_list: List[List[Dict]],
            max_n_tokens: int,
            temperature: float,
            top_p: float = 1.0
    ):
        return [self.generate(conv) for conv in conv_list]


class GPT(LanguageModel):
    API_RETRY_SLEEP = 0
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    # TODO：别忘了删除key
    # client = openai.OpenAI(api_key='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')
    def __init__(self, model_name, api_key):
        self.api_key='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'
        self.model_name='gpt-4-0125-preview'
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(
            self,
            conv: List[Dict],
            max_n_tokens: int,
            temperature: float,
            top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: 生成的回复
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(type(e), e)
                # time.sleep(self.API_RETRY_SLEEP)

            # time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
            self,
            conv_list: List[List[Dict]],
            max_n_tokens: int,
            temperature: float,
            top_p: float = 1.0
    ):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in conv_list]


websocket.enableTrace(False)


class _WsParam(object):
    # 初始化
    def __init__(self, appid, api_key, api_secret, spark_url):
        self.APPID = appid
        self.APIKey = api_key
        self.APISecret = api_secret
        self.host = urlparse(spark_url).netloc
        self.path = urlparse(spark_url).path
        self.Spark_url = spark_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


class Spark(LanguageModel):
    def __init__(self, appid, api_key, api_secret, spark_url, domain, model_name='讯飞星火'):
        super().__init__(model_name)
        self.answer = ''
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.spark_url = spark_url
        self.domain = domain

    def on_error(self, ws, error):
        return
        # print("### error:", error)

    # 收到websocket关闭的处理
    def on_close(self, ws, one, two):
        return

    # 收到websocket连接建立的处理
    def on_open(self, ws):
        thread.start_new_thread(self.run, (ws,))

    def run(self, ws):
        data = json.dumps(self.gen_params(appid=ws.appid, domain=ws.domain, question=ws.question))
        ws.send(data)

    # 收到websocket消息的处理
    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            # print(content, end="")
            self.answer += content
            if status == 2:
                ws.close()

    @staticmethod
    def gen_params(appid, domain, question):
        """
        通过appid和用户的提问来生成请参数
        """
        data = {
            "header": {
                "app_id": appid,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    "domain": domain,
                    "temperature": 0.5,
                    "max_tokens": 2048
                }
            },
            "payload": {
                "message": {
                    "text": question
                }
            }
        }
        return data

    def generate(self, question, *args):
        self.answer = ''

        ws_param = _WsParam(self.appid, self.api_key, self.api_secret, self.spark_url)

        ws_url = ws_param.create_url()
        ws = websocket.WebSocketApp(ws_url, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close,
                                    on_open=self.on_open)
        ws.appid = self.appid
        ws.question = question
        ws.domain = self.domain
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return self.answer

    def batched_generate(
            self,
            conv_list: List[List[Dict]],
            max_n_tokens: int,
            temperature: float,
            top_p: float = 1.0
    ):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in conv_list]
