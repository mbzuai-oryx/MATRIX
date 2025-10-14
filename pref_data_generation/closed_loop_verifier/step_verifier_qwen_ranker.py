from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import  torch
# from tongagent.utils import load_config
import json
import time
import datetime

# using now() to get current time

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenModel():

    def __init__(self, model_name, lora_path=None):
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        # )
        self.model_name = model_name
        if "VL" in model_name:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            # self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.lora_path = lora_path
        # self.model, self.tokenizer = self.load_pretrained_model()
    
        with open('closed_loop_verifier/prompt/rank_verifier_system.prompt', 'r') as file:
            self.system_prompt_ori = file.read()

        with open('closed_loop_verifier/prompt/rank_verifier_user.prompt', 'r') as file:
            self.user_prompt_ori = file.read()        


    def extract_between(self, str1, str2, str3):
        try:
            # Find the starting index of str1 in str3
            start_index = str3.find(str1)
            if start_index == -1:
                return None  # str1 not found in str3
            
            # Adjust the start_index to the end of str1
            start_index += len(str1)
            
            # Find the ending index of str2 in str3, starting from start_index
            end_index = str3.find(str2, start_index)
            if end_index == -1:
                return None  # str2 not found in str3
            
            # Extract the substring between str1 and str2
            return str3[start_index:end_index]
        except Exception as e:
            return None
        
    def forward(self, messages):
       
        print ('QWEN INPUT messages', messages)

        if "VL" in self.model_name:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True        )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                top_k=100,
                do_sample=True,
                repetition_penalty=1.05,
                # num_beams=3, # pengxiang modified
                # num_return_sequences=num_return_sequences # pengxiang modified
            )
            # generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f"=======+++++++ QWen Verifier : {output_text}")
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"=======+++++++ QWen Verifier : {response}")
        return response
    
def write_json(data, filename):
    """
    Write a JSON-compatible Python dictionary to a file.

    :param data: The JSON-compatible dictionary to write.
    :param filename: The name of the file to write to.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

class Step_Verifier():
    def __init__(self, model = "Qwen/Qwen2.5-7B-Instruct"):
         
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     config.verifier.model_path,
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(config.verifier.model_path)
        self.model = model
        if "VL" in self.model:
            self.qwen = QwenModel("Qwen/Qwen2-VL-7B-Instruct")
        else:
            self.qwen = QwenModel("Qwen/Qwen2.5-7B-Instruct")
        with open('closed_loop_verifier/prompt/rank_verifier_system.prompt', 'r') as file:
            self.system_prompt_ori = file.read()

        with open('closed_loop_verifier/prompt/rank_verifier_user.prompt', 'r') as file:
            self.user_prompt_ori = file.read()        


    def extract_between(self, str1, str2, str3):
        try:
            # Find the starting index of str1 in str3
            start_index = str3.find(str1)
            if start_index == -1:
                return None  # str1 not found in str3
            
            # Adjust the start_index to the end of str1
            start_index += len(str1)
            
            # Find the ending index of str2 in str3, starting from start_index
            end_index = str3.find(str2, start_index)
            if end_index == -1:
                return None  # str2 not found in str3
            
            # Extract the substring between str1 and str2
            return str3[start_index:end_index]
        except Exception as e:
            
            return None




    def get_response(self, messages):
        response = self.qwen.forward(messages)

        # print ('response111',response)
        analysis_response = self.extract_between('```json\n', '\n```',response)
        print ('analysis_response', analysis_response)

        try:
            analysis = json.loads(analysis_response)
        except Exception as e:
            print(f"An error occurred in the json_loads of verifier: {e}")
            return 'error'
        
        print ('analysis response', analysis)
        return analysis


    def forward(self, N, task, previous_steps, current_observations, current_steps, images=None, captions = None):
        # N = str(N)
        system_prompt = self.system_prompt_ori.replace('<N>', str(N))
        if captions:
            task = task + '\n' + "For your better understanding of the image content, we provide the captions for the images are as follows:"
            for idx, caption in enumerate(captions):
                task = task + '\n' + "This is the caption for the image " + str(idx+1) + ": " + caption +  '\n'

        usr_prompt = self.user_prompt_ori.replace('<task>', task)

        eval_set = {}
        for i in range(N):
            eval_set["PREVIOUS_RESULT"] =  previous_steps
            eval_set['Trajectory' + str(i+1)] = {
                "CURRENT_STEP": current_steps[i],
                "CURRENT_RESULT": current_observations[i]
            }
        eval_set = json.dumps(eval_set, indent=4)
        usr_prompt = usr_prompt.replace('<step_set>', eval_set)
        if images and "VL" in self.model:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":  [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": usr_prompt}]}
            ]   
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_prompt}
            ]
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        
        # QwenVL2 template
        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content":  {"type": "text", "text": usr_prompt}}
        # ] 
        score= self.get_response(messages)

        messages.append({"role": "verifier", "content": score})
        write_json(messages, f'/home/lipengxiang/codes/DPOagent/TongAgent/closed_loop_verifier/plauground/verifier-messages-{current_time}.json')
        # print('Verifier Result:', score)

        return score
        



def main():
    verifier = Step_Verifier("Qwen/Qwen2.5-7B-Instruct")
    # verifier.get_response()
    previous_steps = ["no previous step is none" * 3]
    current_steps = ["Thought: First, I need to find out the price of the NVIDIA GeForce RTX 4070 SUPER. I will use the `visualizer` tool to extract the price information from the NVIDIA's official website.\nCode:\n```py\nprice_info = visualizer(question=\"What is the price of the NVIDIA GeForce RTX 4070 SUPER?\", image_path=\"gta_dataset/image/image_14.jpg\")\nprint(price_info)\n```", "Thought: I will use the `visualizer` tool to get the price of the NVIDIA GeForce RTX 4070 SUPER from NVIDIA's official website. Then I will calculate the total cost by multiplying the price by the number of GPUs. Finally, I will use the `final_answer` tool to provide the final answer.\nCode:\n```py\nprice = visualizer(question=\"How much does the NVIDIA GeForce RTX 4070 SUPER cost?\", image_path=\"gta_dataset/image/image_14.jpg\")\nprint(price)\n```","Thought: I need to find the price of one NVIDIA GeForce RTX 4070 SUPER on NVIDIA's official website in January. I will use the `ask_search_agent` tool to search for this information.\nCode:\n```py\n# Search for the price of the NVIDIA GeForce RTX 4070 SUPER on NVIDIA's official website in January\nsearch_result = ask_search_agent(query=\"NVIDIA GeForce RTX 4070 SUPER price January\")\nprint(search_result)\n```"]

    current_observations = ['The NVIDIA GeForce RTX 4070 SUPER is priced at $299.', 'The NVIDIA GeForce RTX 4070 SUPER is priced at $399.', 'The NVIDIA GeForce RTX 4070 SUPER is priced at $699.', 'The NVIDIA GeForce RTX 4070 SUPER costs $699.', 'The NVIDIA GeForce RTX 4070 SUPER is priced at $599.', 'The NVIDIA GeForce RTX 4070 SUPER is priced at $399.',"Code execution failed due to the following error:\nConnection error."]

    # current_observations = [current_observations[1], current_observations[0], current_observations[2]]
    # current_steps = [current_steps[1], current_steps[0], current_steps[2]]
    task = "The men in the picture want to buy one NVIDIA GeForce RTX 4070 SUPER each. According to NVIDIA's official website in January, how many dollars will they need to spend in total?\nAttachement: gta_dataset/image/image_14.jpg"
    # verifier.forward(3, task, previous_steps, current_observations, current_steps, images = ["gta_dataset/image/image_14.jpg"],captions=["A photo of 4070"])
    verifier.forward(3, task, previous_steps, current_observations, current_steps, captions=["A photo of 4070"])

if __name__ == "__main__":
    main()
