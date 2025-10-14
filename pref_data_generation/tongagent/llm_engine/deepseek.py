import os
import sys
sys.path.append('/home/lipengxiang/codes/DPOagent/TongAgent')
from openai import AzureOpenAI, RateLimitError, OpenAI
 
# from openai import AzureOpenAI, RateLimitError, OpenAI
from transformers.agents.llm_engine import MessageRole, get_clean_message_list, HfApiEngine
from tongagent.utils import load_config
import base64
import re

 

def get_deepseek_open_ai_client(model="deepseek"):
    config = load_config()
    deepseek_config = getattr(config, model)
    

    client = OpenAI(

    api_key=f"{deepseek_config.api_key}",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
    return client, deepseek_config.model_name 

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class DeepSeekEngine(HfApiEngine):
    def __init__(self, model="deepseek"):
  
        client, model_name = get_deepseek_open_ai_client(model)
        self.model_name= model_name
        self.client = client
        
    def __call__(self, messages, stop_sequences=[], *args, **kwargs) -> str:
        # print ('----------------raw message',messages)
        
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
 
        retry = 3
        for i in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stop=stop_sequences,
                )
                break
            except RateLimitError as e:
                print("catch rate limit error")
                import time
                time.sleep(10)
            
        # raise Exception(response)
        if 'r1' in self.model_name:
            print("Reasnoing process:")
            print(response.choices[0].message.reasoning_content)
 
        print("Deepskeek final answer:")
        print(response.choices[0].message.content)
        # print(response.choices[0].message.content)
        return response.choices[0].message.content

def main():
    # Example usage
    deepseek = DeepSeekEngine()
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]
    response = deepseek(messages)
    print(response)
if __name__ == "__main__":
    main()