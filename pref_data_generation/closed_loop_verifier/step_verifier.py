from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

from tongagent.utils import load_config
import json


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
    def __init__(self):
        config = load_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            config.verifier.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.verifier.model_path)

        with open(config.verifier.system_prompt_path, 'r') as file:
            self.system_prompt_ori = file.read()

        with open(config.verifier.user_prompt_path, 'r') as file:
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
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=15000
            )
        except Exception as e:
            print(f"An error occurred in the generation of verifier: {e}")
            return 'error'

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
    

    def traj_process(self, traj):

        question=traj[0]["content"]
        question=question[question.find('Task: ')+len('Task: '):]

        all_step=len(traj)
        # print ('all_step', all_step)
        start = 2
        end = all_step-1  # You can set this to any number you want to stop at

        previous_observation='None'
        for step in range(start, end, 2):  # The range function includes the start value and excludes the stop value
            # print('step', step)
            if step ==2:
                previous_observation=traj[step]['content'] + '\n'
            else:
                previous_observation=previous_observation+traj[step]['content'] + '\n'

        current_observation=traj[end]['content']

        return question, previous_observation, current_observation


    def message_observation(self, traj):

        question, previous_observation, current_observation = self.traj_process(traj)

        user_prompt = self.user_prompt_ori.replace('<task>', question)
        user_prompt = user_prompt.replace('<previous results>', previous_observation)
        user_prompt = user_prompt.replace('<current result>' , current_observation)

        print ('=======================================================')
        print ('11111   question:::',question)
        print ('22222   previous_observation:::',previous_observation)
        print ('33333   current_observation:::',current_observation)

        print ('++++++++++++++++++++++user_prompt',user_prompt)

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": self.system_prompt_ori + '\n' + user_prompt}
        ]      
        # messages = [
        #     {"role": "system", "content": self.system_prompt_ori},
        #     {"role": "user", "content": user_prompt}
        # ]          
        return messages

    def forward(self,traj):
        messages = self.message_observation(traj)
        score= self.get_response(messages)
        return score



def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    return data


verifier = Step_Verifier()

traj_path='closed_loop/zip_json/qwen_tuned.json'
traj=read_json(traj_path)
traj_num=len(traj)
step_num=100

eval_count=0
score_count=0

score_list=["0","1","2","3","4","5","6","7","8","9","10"]
score_num_dict={}
for t in score_list:
    score_num_dict[t]=0

for i in range (traj_num):
    t=traj[i]["conversations"]
    t=t[1:]
    traj_len=len(t)
    for j in range (3, traj_len, 2):
        print ('-------------------traj_num:', i, 'traj_len:', j)
        # print ('----- previous', t[:j-1])
        # print ('----- current', t[j-1])
        output=verifier.forward(t[:j])
        print('verifier:', output)
        try:
            score= int(output["Score"])
            print('parse correct, Score:', score)
        except:
            score= 0
            print('parse incorrect, Score:', score)   

        score_num_dict[str(score)] = score_num_dict[str(score)]+1

        score_count=score_count+score
        eval_count=eval_count+1
        
        print ('score_count:', score_count)
        print ('eval_count:', eval_count)
        print ('average num:', score_count/eval_count)
        write_json(score_num_dict, traj_path[:-5]+'_score.json')

    # if i > step_num:
    #     break
              