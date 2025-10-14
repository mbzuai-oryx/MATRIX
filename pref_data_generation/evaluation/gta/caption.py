from openai import AzureOpenAI
import base64
REGION = "eastus"
MODEL = "gpt-4o-mini-2024-07-18"
API_KEY = "92209c51c76bdbf120a5eee1847c4f3b"

API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
ENDPOINT = f"{API_BASE}/{REGION}"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
)



# call gpt with encoded image
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        
        {"role": "user", "content": 
            [
                {"type": "text", "text": "Caption the image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image('data/gta_dataset/image/image_289.jpg')}"}}
            ]
        }
    ],
)

print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)