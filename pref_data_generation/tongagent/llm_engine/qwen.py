import sys
sys.path.append('/home/lipengxiang/codes/DPOagent/TongAgent')

from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list
from tongagent.utils import load_config
import re

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info
from openai import OpenAI

import torch

def load_pretrained_model(model_name):
    torch.manual_seed(0)
    print("from pretrained", model_name)
    if "VL" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_client_model(endpoint):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{endpoint}:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

class ModelSingleton():
    def __new__(cls, model_name, lora_path=None):
        if hasattr(cls, "model_dict") and model_name in cls.model_dict:
            return cls

        if not hasattr(cls, "model_dict"):
            cls.model_dict = dict()
            
        if "VL" in model_name:
            model, tokenizer = load_pretrained_model(model_name)
            if lora_path is not None:
                print("Load Qwen-VL from lora", lora_path)
                import time
                from peft.peft_model import PeftModel
                time.sleep(10)
                model = PeftModel.from_pretrained(model, lora_path)
                model.merge_and_unload()
            cls.model_dict[model_name] = (model, tokenizer)
            
        else:
            config = load_config()
            model = load_client_model(config.qwen.endpoint)
            tokenizer = None
            cls.model_dict[model_name] = (model, tokenizer)
        return cls

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
    # MessageRole.SYSTEM: MessageRole.USER
}

from typing import Optional
class QwenEngine(HfApiEngine):
    def __init__(self, model_name: str = "", lora_path: Optional[str] = None, beam_size=5):
        module = ModelSingleton(model_name, lora_path)
        self.has_vision = False
        model, tokenizer = module.model_dict[model_name]
        if 'VL' in model_name:
            self.has_vision = True
            self.processor = tokenizer # for VLM use processor as tokenizer
            
        self.model, self.tokenizer = model, tokenizer
        self.model_name = model_name
        self.beam_size = beam_size
    def call_llm(self, messages, stop_sequences=[], *args, **kwargs):
        assert not self.has_vision, "Should use this function with Qwen LLM"
        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=512,
        #     temperature=0.7,
        #     top_p=0.8,
        #     top_k=100,
        #     do_sample=True,
        #     repetition_penalty=1.05
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]

        # answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return answer
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            n = 3 # pengxiang modified
        )        
        # return response.choices[0].message.content
        return response

    def call_vlm(self, messages, stop_sequences=[], *args, **kwargs):
        print("call vlm")
        assert self.has_vision, "Should use this function with Qwen VL model"
        image_paths = kwargs.get("image_paths", [])
        beam_size = kwargs.get("beam_size", 1)
        if beam_size > 1:
            for msg_id, msg in enumerate(messages):
                if msg["role"] == "user":
                    content_replace = []
                    if len(image_paths) == 1:
                        for image_path in image_paths:
                            content_replace.append({
                                "type": "image",
                                "image": image_path
                                # "min_pixels": 112896,
                                # "max_pixels": 112896
                            })
                        
                    if len(image_paths) > 1:
                        for image_path in image_paths:
                            content_replace.append({
                                "type": "image",
                                "image": image_path,
                                "min_pixels": 100 * 28 * 28,
                                "max_pixels": 512 * 28 * 28
                            })
                        
                    content = {"type": "text", "text": msg["content"]}
                    content_replace.append(content)
                    messages[msg_id] = {
                        "role": "user",
                        "content": content_replace
                    }
                    break
        else:
            for msg_id, msg in enumerate(messages):
                if msg["role"] == "user":
                    content_replace = []
                    if len(image_paths) > 0:
                        for image_path in image_paths:
                            content_replace.append({
                                "type": "image",
                                "image": image_path
                            })
                        
                    content = {"type": "text", "text": msg["content"]}
                    content_replace.append(content)
                    messages[msg_id] = {
                        "role": "user",
                        "content": content_replace
                    }
                    break
            
        print("msg=", messages)
        text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        num_return_sequences = beam_size
        # generated_ids = self.model.generate(
        #     **inputs, 
        #     max_new_tokens=512,
        #     temperature=1.2,  # Slightly lower for balanced randomness
        #     top_p=0.9,        # Broader sampling while maintaining coherence
        #     top_k=50,         
        #     do_sample=True,   # Enables stochastic sampling for diversity
        #     num_beams=6,      
        #     num_beam_groups=3,  # Increase groups for more diverse beams
        #     diversity_penalty=0.7,  # Higher penalty for greater diversity
        #     repetition_penalty=1.05,  # Avoid repetitive outputs
        #     num_return_sequences=num_return_sequences
        # )

    #     generated_ids = self.model.generate(
    #     **inputs, 
    #     max_new_tokens=512,
    #     temperature=1.5,          # Default for deterministic beam search
    #     top_p=1.0,                # No nucleus sampling in beam search
    #     top_k=50,                 # Optional, but has minimal impact in this mode
    #     do_sample=False,          # Required for group beam search
    #     num_beams=15,              
    #     num_beam_groups=5,        # Increase groups for diverse beams
    #     diversity_penalty=0.7,    # Encourage more diverse beams
    #     num_return_sequences=num_return_sequences
    # )
        if num_return_sequences > 1:
            generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=1.2,          # Higher for randomness
            top_p=0.8,                # Use nucleus sampling for diversity
            top_k=100,                 # Limits token sampling
            do_sample=True,           # Enables stochastic sampling
            # num_beams=1,              # No beam search when sampling
            # diversity_penalty=0.0,    # Not applicable for sampling
            num_return_sequences=num_return_sequences
        )
        if num_return_sequences == 1:
            generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=100,
            do_sample=True,
            repetition_penalty=1.05,
            num_return_sequences=num_return_sequences
        )
        # new generate
        # generated_ids = self.model.generate(
        #     **inputs, 
        #     max_new_tokens=1024,
        #     do_sample=True,
        #     num_beams=3, # pengxiang modified
        #     num_return_sequences=num_return_sequences # pengxiang modified
        # )


        inputs.input_ids = inputs.input_ids.repeat(num_return_sequences, 1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("output_text=", output_text)

        # return output_text[0]
        return output_text
    
    def __call__(self, messages, stop_sequences=[], *args, **kwargs) -> str:
        # print ('----------------raw message',messages)
        torch.cuda.empty_cache()
        image_paths = kwargs.get("image_paths", [])
        beam_size = kwargs.get("beam_size", 1)
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        #print ('----------------processed message',messages)
        task = messages[0]
        msgs = []
        for msg in messages:
            # print(msg["role"].value)
            if msg["role"] == MessageRole.SYSTEM:
                msgs.append(
                    {
                        "role": "system",
                        "content": msg["content"]
                    }
                )
            else:
                msgs.append(
                    {
                        "role": "user" if msg["role"] == MessageRole.USER else "assistant",
                        "content": msg["content"]
                    }
                )
        if not self.has_vision:
            answers = self.call_llm(messages, stop_sequences=stop_sequences)
        else:
            answers = self.call_vlm(messages, stop_sequences=stop_sequences, image_paths=image_paths, beam_size=beam_size)
        print(answers)
        new_answers = []
        for ansidx, answer in enumerate(answers):
            for stop in stop_sequences:
                stop_idx = answer.find(stop)
                if stop_idx == -1:
                    continue
                answer = answer[:stop_idx]
                new_answers.append(answer)
            if len(new_answers) == ansidx:
                new_answers.append(answer)
        
        if beam_size == 1:
            return new_answers[0]
        return new_answers
        
if __name__ == "__main__":
    # model, tokenizer = load_pretrained_model("Qwen/Qwen2-VL-7B-Instruct")

    # print(model)
    qwen = QwenEngine("Qwen/Qwen2-VL-7B-Instruct")
    # qwen = QwenEngine("Qwen/Qwen2-VL-2B-Instruct")
    msgs = [
{
                    "role": "system",
                    "content": "You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.\nTo do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.\nTo solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.\n\nAt each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.\nThen in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_action>' sequence.\nDuring each intermediate step, you can use 'print()' to save whatever important information you will then need. DO NOT generate a code which does not call 'print()' because you will lose this information. You can assume all tools must have a return that can be printed. \nThese print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.\nYou will save all intermediate file outputs to a folder by the relative path '.cache'.\nIn the end you have to return a final answer using the `final_answer` tool. \n\nHere are a few examples using notional tools:\n\n---\nTask: \"What is the result of the following operation: 5 + 3 + 1294.678?\"\n\nThought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool\n\nCode:\n```py\nresult = 5 + 3 + 1294.678\nfinal_answer(result)\n```<end_action>\n\n---\nTask: \"Which city has the highest population: Guangzhou or Shanghai?\"\n\nThought: I need to get the populations for both cities and compare them: I will use the tool `ask_search_agent` to get the population of both cities.\nCode:\n```py\npopulation_guangzhou = ask_search_agent(\"Guangzhou population\")\nprint(\"Population Guangzhou:\", population_guangzhou)\npopulation_shanghai = ask_search_agent(\"Shanghai population\")\nprint(\"Population Shanghai:\", population_shanghai)\n```<end_action>\nObservation:\nPopulation Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']\nPopulation Shanghai: '26 million (2019)'\n\nThought: Now I know that Shanghai has the highest population.\nCode:\n```py\nfinal_answer(\"Shanghai\")\n```<end_action>\n\n---\nTask: \"What is the current age of the pope, raised to the power 0.36?\"\n\nThought: I will use the tool `ask_search_agent` to get the age of the pope, then raise it to the power 0.36.\nCode:\n```py\npope_age = ask_search_agent(query=\"current pope age\")\nprint(\"Pope age:\", pope_age)\n```<end_action>\nObservation:\nPope age: \"The pope Francis is currently 85 years old.\"\n\nThought: I know that the pope is 85 years old. Let's compute the result using python code.\nCode:\n```py\npope_current_age = 85 ** 0.36\nfinal_answer(pope_current_age)\n```<end_action>\n\n---\nTask: \"Convert the table into a pie chart. Attachment: .cache/table1.jpg\"\n\nThought: To convert the table in the image to a pie chart, I will first need to extract the text from the image to get the data that is presented in the table. I will use the `visualizer` tool to analyze the image and extract the textual information in the table format. After that, I can process the data and create a pie chart using a plotting library. I'll start by extracting the text from the image of the table.\nCode:\n```py\ntable_data = visualizer(question=\"Please extract the data from the table in the image.\", image_path=\".cache/table1.jpg\")\nprint(table_data)\n```<end_action>\nObservation: Here is the extracted data from the table:\n| PRODUCT NAME | SOLD AMOUNT |\n|----------|----------|\n| PRODUCT1   | 500      |\n| PRODUCT2   | 501      |\n| PRODUCT3   | 502      |\n| PRODUCT4   | 503      |\n| PRODUCT5   | 504      |\n| PRODUCT6   | 0        |\n| PRODUCT7   | 0        |\n| PRODUCT8   | 0        |\n| PRODUCT9   | 508      |\n| PRODUCT10  | 509      |\n| PRODUCT11  | 510      |\n\nThought: Now that I have extracted the data from the table, I will process it to extract the product names and their respective sales totals. Using this data, I can then create a pie chart to visually represent the sales distribution among the different products. First, I will organize the data into a suitable format, and then I will use the `matplotlib` library to create the pie chart.\nCode:\n```py\nimport matplotlib.pyplot as plt\n# Data extracted from the table\nproducts = [\"PRODUCT1\", \"PRODUCT2\", \"PRODUCT3\", \"PRODUCT4\", \"PRODUCT5\", \"PRODUCT6\", \"PRODUCT7\", \"PRODUCT8\", \"PRODUCT9\", \"PRODUCT10\", \"PRODUCT11\"]\nsales = [500, 501, 502, 503, 504, 0, 0, 0, 508, 509, 510]\n# Creating a pie chart\nplt.figure(figsize=(10, 7))\nplt.pie(sales, labels=products, autopct='%1.1f%%', startangle=140)\nplt.title(\"Sales Distribution by Product\")\nplt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.\n# Saving the pie chart to a file\nchart_path = '.cache/sales_distribution_pie_chart.jpg'\nplt.savefig(chart_path)\nplt.close()\nprint(\"Pie chart saved at:\", chart_path)\n```<end_action>\nObservation: Pie chart saved at: .cache/sales_distribution_pie_chart.jpg\n\nThought: The pie chart representing the sales distribution by product has been successfully created and saved. Now, I will use the `final_answer` tool to provide the path to the saved pie chart as the final output.\nCode:\n```py\nfinal_answer(\"Pie chart saved at: data/tongagent/sales_distribution_pie_chart.jpg\")\n```<end_action>\n\n---\nTask: \"Identify and list the types of fruits visible in this image. Attachment: .cache/000000202178.jpg\"\n\nThought: I will use the `objectlocation` tool to identify and list the types of fruits visible in the provided image. This tool will help localize different fruits present in the image, and I can then compile the identified types.\nCode:\n```py\nfruit_types = objectlocation(object=\"fruit\", image_path=\".cache/000000202178.jpg\")\nprint(fruit_types)\n```<end_action>\nObservation: [[173.91, 2.34, 372.53, 87.86], [170.28, 2.32, 398.48, 121.89], [410.71, 42.45, 483.26, 130.54]]\n\n\nThought: Now, I have found bounding boxes of fruits. I will crop these regions of fruits and save in new files.\nCode:\n```py\nfrom PIL import Image\nimport os\n\nimage_path = \".cache/000000202178.jpg\"\nimage = Image.open(image_path)\n\nroot = \".cache/output\"\nos.makedirs(root, exist_ok=True)\ncount = 0\nfor bbox in fruit_types:\n   crop_image = image.crop(bbox)\n   crop_image.save(f'{root}/{count}.jpg')\n   print(f'{root}/{count}.jpg')\n   count = count+1\n```<end_action>\nObservation: .cache/output/0.jpg, .cache/output/1.jpg, .cache/output/2.jpg,\n\nThought: I will list all the images in the folder '.cache/output', then apply the `visualizer` tool to each image for the types of fruits.\nCode: \n```py\nimage_folder_path = '.cache/output'\nimage_files = [file for file in os.listdir(image_folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]\nfor image_file in image_files:\n    image_path = os.path.join(image_folder_path, image_file)\n    fruit_type = visualizer(question=\"What types of fruits are visible in this image?\", image_path=image_path)\n    print(fruit_type)\nObservation: Pineapple\nBananas\nMango\n```<end_action>\n\nThought: I have identified the types of fruits present in the image. Now, I will compile the list of fruits and return it as the final answer.\nCode:\n```py\nfruit_list = [\n    \"Pineapple\",\n    \"Bananas\",\n    \"Mango\"\n]\nfinal_answer(fruit_list)\n```<end_action>\n\nAbove example were using notional tools that might not exist for you. You only have access to those tools:\n\n\n- visualizer: A tool that can answer questions about attached images.\n    Takes inputs: {'question': {'description': 'the question to answer', 'type': 'string'}, 'image_path': {'description': 'The path to the image on which to answer the question', 'type': 'string'}}\n    Returns an output of type: string\n\n- facedetection: A tool that can detect human faces in given images, outputing the bounding boxes of the human faces.\n    Takes inputs: {'image_path': {'description': 'The path to the image on which to localize objects. This should be a local path to downloaded image.', 'type': 'string'}}\n    Returns an output of type: any\n\n- objectlocation: A tool that can localize objects in given images, outputing the bounding boxes of the objects.\n    Takes inputs: {'object': {'description': 'the object that need to be localized', 'type': 'string'}, 'image_path': {'description': 'The path to the image on which to localize objects. This should be a local path to downloaded image.', 'type': 'string'}}\n    Returns an output of type: any\n\n- inspect_file_as_text: You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it. This tool handles the following file extensions: [\".html\", \".htm\", \".xlsx\", \".pptx\", \".wav\", \".mp3\", \".flac\", \".pdf\", \".docx\"], and all other types of text files. IT DOES NOT HANDLE IMAGES.\n    Takes inputs: {'question': {'description': '[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.', 'type': 'string'}, 'file_path': {'description': \"The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT USE THIS TOOL FOR A WEBPAGE: use the search tool instead!\", 'type': 'string'}}\n    Returns an output of type: string\n\n- image_generator: This is a tool that creates an image according to a prompt, which is a text description.\n    Takes inputs: {'prompt': {'type': 'string', 'description': \"The image generator prompt. Don't hesitate to add details in the prompt to make the image look better, like 'high-res, photorealistic', etc.\"}}\n    Returns an output of type: any\n\n- segmentation: A tool that can do instance segmentation on the given image.\n    Takes inputs: {'image_path': {'description': 'The path of image that the tool can read.', 'type': 'string'}, 'prompt': {'description': \"The bounding box that you want this model to segment. The bounding boxes could be from user input or tool `objectlocation`. You can set it as None or empty list to enable 'Segment Anything' mode.\", 'type': 'any'}}\n    Returns an output of type: string\n\n- image_edit: A tool that can edit image based on the user prompt. Return a file path for printing.\n    Takes inputs: {'prompt': {'description': 'The user prompt that instruct how to edit the image.', 'type': 'string'}, 'image_path': {'description': 'The image path that this tool will try to edit.', 'type': 'string'}}\n    Returns an output of type: string\n\n- ask_search_agent: This will send a message to a team member that will browse the internet to answer your question. Ask him for all your web-search related questions, but he's unable to do problem-solving. Provide him as much context as possible, in particular if you need to search on a specific timeframe! And don't hesitate to provide them with a complex search task, like finding a difference between two webpages.\n    Takes inputs: {'query': {'description': \"Your question, as a natural language sentence with a verb! You are talking to an human, so provide them with as much context as possible! DO NOT ASK a google-like query like 'paper about fish species 2011': instead ask a real sentence like: 'What appears on the last figure of a paper about fish species published in 2011?'\", 'type': 'string'}}\n    Returns an output of type: string\n\n- final_answer: Provides a final answer to the given problem.\n    Takes inputs: {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}\n    Returns an output of type: any\n\nYou also can perform computations in the Python code that you generate.\n\nHere are the rules you should always follow to solve your task:\n1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_action>' sequence, else you will fail.\n2. Use only variables that you have defined!\n3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = ask_search_agent({'query': \"What is the place where James Bond lives?\"})', but use the arguments directly as in 'answer = ask_search_agent(query=\"What is the place where James Bond lives?\")'.\n4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.\n5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.\n6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.\n7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.\n8. You can use imports in your code, but only from the following list of modules: ['sklearn', 'statistics', 'sympy', 'collections', 'chess', 'random', 'pptx', 'PyPDF2', 'cv2', 'xml', 'os', 'math', 'pandas', 'requests', 'stat', 'itertools', 'fractions', 'bs4', 'datetime', 'yahoo_finance', 're', 'PIL', 'scipy', 'pydub', 'unicodedata', 'pickle', 'Bio', 'torch', 'pubchempy', 'time', 'matplotlib', 'numpy', 'zipfile', 'json', 'io', 'queue', 'csv']\n9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.\n10. Don't give up! You're in charge of solving the task, not providing directions to solve it.\n\nNow Begin! If you solve the task correctly, you will receive a reward of $1,000,000.\n"
                },
                {
                    "role": "user",
                    "content": [       
                        {
                            "type": "image",
                            "image": "tests/data/gta_case1.png",
                        },
                        { 
                            "type": "text",
                            "text": "Task: Which country won the gold medal of the mixed double game in that year's Olympic Game as shown in the image?\nAttachement: tests/data/gta_case1.png"
                        }]
                }
                
    ]  
    result = qwen(msgs, stop_sequences = ["<end_action>"])
    print("-------------------------------------------------------------")
    for idx, i in enumerate(result):
        print()
        print()
        print("#############################################################################")
        print(f"                                       {idx}                                  ")
        print("#############################################################################")
        print(i)
    # print(result)
    