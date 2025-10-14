from typing import Callable, List, Optional, Dict, Any
import os
import json
import shutil

# from tongagent.tools.tool_box import get_visual_model_tool_box, get_visual_model_tool_box_for_gaia
from tongagent.tools.tool_box import get_toolbox_for_dpo as get_visual_model_tool_box
from tongagent.llm_engine.gpt import TongGPTEngine
from tongagent.prompt import DEFAULT_REACT_CODE_SYSTEM_PROMPT
from tongagent.utils import load_config, Logger
config = load_config()
search_config = getattr(config, 'search_agent')
if search_config.type =='api':
    print ('use the gpt4o mini api as the search tool')
    from tongagent.agents.search_agent_api import SearchTool
else:
    print ('use the search agent as the search tool')
    from tongagent.agents.search_agent import SearchTool

from tongagent.utils import gen_random_id, CACHE_FOLDER

from transformers.agents import ReactCodeAgent, HfApiEngine
from transformers.agents.agents import AgentGenerationError, AgentParsingError, AgentError, AgentMaxIterationsError, parse_code_blob, BASE_PYTHON_TOOLS, AgentExecutionError
# from transformers.agents.prompts import DEFAULT_REACT_CODE_SYSTEM_PROMPT
from transformers.agents.tools import DEFAULT_TOOL_DESCRIPTION_TEMPLATE, Tool
from transformers.agents.python_interpreter import evaluate_python_code, LIST_SAFE_MODULES
# from closed_loop_verifier.step_verifier_gpt_ranker import Step_Verifier
from closed_loop_verifier.step_verifier_qwen_ranker import Step_Verifier

from transformers.agents.llm_engine import  MessageRole
class AgentToleranceError(AgentError):
    pass

SAFE_MODULES = list(set(LIST_SAFE_MODULES + [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "csv",
    "fractions",
    "matplotlib",
    "pickle",
    "cv2"
]))



def evaluate_python_code_modify(
    code: str,
    static_tools: Optional[Dict[str, Callable]] = None,
    custom_tools: Optional[Dict[str, Callable]] = None,
    state: Optional[Dict[str, Any]] = None,
    authorized_imports: List[str] = SAFE_MODULES,
):
    # print('authorized_imports', authorized_imports)
    result = evaluate_python_code(
        code,
        static_tools,
        custom_tools,
        state,
        authorized_imports
    )
    if state is not None and "print_outputs" in state and type(state["print_outputs"]) is str:
        state["print_outputs"] = state["print_outputs"] if len(state["print_outputs"]) > 0 else "No observation found from the code execution. You should use `print` function if need some information from the code execution."
    return result

class DPOAgent(ReactCodeAgent):
    def __init__(
        self, 
        tools: List[Tool], 
        llm_engine: Callable = HfApiEngine(), 
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT, 
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE, 
        additional_authorized_imports: List[str] | None = None, 
        planning_interval: int | None = None,
        error_tolerance_count: int = -1, 
        **kwargs):

        super().__init__(tools=tools, llm_engine=llm_engine, system_prompt=system_prompt, tool_description_template=tool_description_template, additional_authorized_imports=additional_authorized_imports, planning_interval=planning_interval, **kwargs)
        self.captions = None 
        self.image_paths = None
        self.python_evaluator = evaluate_python_code_modify
        self.error_tolerance_count = error_tolerance_count
        self.beam_size  = kwargs.get("sample_size", 5)
        verifier = kwargs.get("verifier", 'best_selector')

        
        self.step_verifier = Step_Verifier("Qwen/Qwen2.5-7B-Instruct")
        # self.authorized_imports = SAFE_MODULES

        # create save folder and logger
        self.save_folder_id = gen_random_id()           
        path = os.path.join(CACHE_FOLDER+'_qwen_7b_failed', self.save_folder_id)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.dpo_logger = Logger(os.path.join(path, "dpo_agent.log"), level='debug')
        self.dpo_logger.logger.info(f"New DPOAgent created with save_folder_id: {self.save_folder_id}")
        self.task_tmp = None
        # self.final_answers = []
         
    def direct_run(self, task: str):
        self.task_tmp = task
        """
        Runs the agent in direct mode, returning outputs only at the end: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        error_count = 0 
        while final_answer is None and iteration < self.max_iterations:
            if self.error_tolerance_count > 0 and error_count == self.error_tolerance_count:
                break
            try:
                if self.planning_interval is not None and iteration % self.planning_interval == 0:
                    self.planning_step(task, is_first_step=(iteration == 0), iteration=iteration)
                step_logs = self.step()
                if "final_answer" in step_logs:
                    final_answer = step_logs["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
                error_count += 1
            finally:
                iteration += 1

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
        elif final_answer is None and error_count == self.error_tolerance_count:
            error_message = f"Reached max execution exception. Max exception tolerance: {self.error_tolerance_count}."
            final_step_log = {"error": AgentToleranceError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
            
        return final_answer
    
    def set_image_paths(self, image_paths: List[str]):
        self.image_paths = image_paths
        
    def set_file_paths(self, file_paths: List[str]):
        self.file_paths = file_paths
            
    def set_captions(self, captions: List[str]):
        self.captions = captions    
    
    
    # def step_verifier(self, observations):
    #     return 0
  
    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()
        self.prompt = agent_memory.copy()
        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])

        try:
            # record the beam size into the logs
            current_step_logs["beam_size"] = self.beam_size

            # Generate LLM outputs with beam search
            llm_outputs = []
            llm_outputs.append(self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"], image_paths=self.image_paths, beam_size=1))
            if self.beam_size > 1:
                llm_outputs = llm_outputs + self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"], image_paths=self.image_paths, beam_size=self.beam_size - 1)
            
            for j in range(len(llm_outputs), self.beam_size):
                llm_outputs.append("LLM output failed. No output generated.") 
            # record the llm outputs into the logs
            for idx, llm_output_single in enumerate(llm_outputs):
                current_step_logs[f"llm_output_{idx}"] = llm_output_single
            current_step_logs[f"llm_outputs"] = llm_outputs

        except Exception as e:
            current_step_logs["beam_size"] = 0
            raise AgentGenerationError(f"Error in generating llm output: {str(e)}.")
        
        self.dpo_logger.logger.info(llm_outputs)
 
        self.dpo_logger.logger.info(f"LLM output number: {len(llm_outputs)}")
        # observations = []
        for idx, llm_output in enumerate(llm_outputs):
            self.logger.debug(f"===== Output {idx} message of the LLM: =====")
            self.logger.debug(llm_output)
            # current_step_logs[f"llm_output_{idx}"] = llm_output
            self.dpo_logger.logger.info(f"===== Output {idx} message of the LLM: =====")
            self.dpo_logger.logger.info(llm_output)


            # Parse
            self.logger.debug(f"===== Extracting {idx} action =====")
            try:
                rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
            except Exception as e:
                error_msg = f"Error in extracting action, trying to parse the whole output. Error trace: {str(e)}"
                self.logger.debug(error_msg)
                rationale, raw_code_action = llm_output, llm_output
                current_step_logs[f"observation_{idx}"] = error_msg

            self.dpo_logger.logger.info(f"REASON {idx}: {rationale}")
            self.dpo_logger.logger.info(f"CODE {idx}: {raw_code_action}")  
            try:
                code_action = parse_code_blob(raw_code_action)
            except Exception as e:
                error_msg = f"Error in code parsing: {str(e)}. Make sure to provide correct code"
                code_action = raw_code_action
                current_step_logs[f"observation_{idx}"] = error_msg
                self.dpo_logger.logger.debug(f"CODE PARSING ERROR: {error_msg}")
                # raise AgentParsingError(error_msg)
                pass

            current_step_logs[f"rationale_{idx}"] = rationale
            current_step_logs[f"tool_call_{idx}"] = {"tool_name": "code interpreter", "tool_arguments": code_action}

            # Execute
            self.log_rationale_code_action(rationale, code_action)
            try:
                self.logger.info(f'authorized_imports {self.authorized_imports}')
                result = self.python_evaluator(
                    code_action,
                    static_tools={
                        **BASE_PYTHON_TOOLS.copy(),
                        **self.toolbox.tools,
                    },
                    custom_tools=self.custom_tools,
                    state=self.state,
                    authorized_imports=self.authorized_imports,
                )
                information = self.state["print_outputs"]
                if "final_answer" in str(code_action) and "You should use `print` fu" in str(information):
                    information = "Final Answer: " + result
                
                self.logger.warning(f"Print outputs_{idx}:")
                self.logger.log(32, information)
                if type(information) is list:
                    information = information[0]
                current_step_logs[f"observation_{idx}"] = information
                self.dpo_logger.logger.info(f"OBSERVATION {idx}: {information}")
                # if type(information) is list:       
                #     observations.append(information[0])
                # else:
                #     observations.append(information)

            except Exception as e:
                error_msg = f"Code execution failed due to the following error:\n{str(e)}"
                if "'dict' object has no attribute 'read'" in str(e):
                    error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
                # if not current_step_logs[f"observation_{idx}"]:
                current_step_logs[f"observation_{idx}"] = error_msg
                self.dpo_logger.logger.info(f"OBSERVATION ERROR {idx}: {error_msg}")
                # observations.append(error_msg)
                pass
                # raise AgentExecutionError(error_msg)
            # self.dpo_logger.logger.info(f"OBSERVATION: {information}")

            
        ###################### BEAM SEARCH FOE THE BEST STEP #####################    
        # ## beam search for the best step
        observations = [current_step_logs[f"observation_{i}"] for i in range(self.beam_size)]
        current_step_logs['observations'] = observations
        if len(self.logs) > 2:
            try:    
                previous_steps = [f"STEP {i}: \n Observation: {self.logs[i]['observation']}\n" for i in range(1, len(self.logs) - 1)] 
                # previous_steps = [f"STEP {i}: {self.logs[i]['llm_output']} \n Observation: {self.logs[i]['observation']}\n" for i in range(1, len(self.logs) - 1)] 
                # previous_steps = [f"Sample {i} Observation: {self.logs[i]['observation']}\n" for i in range(1, len(self.logs) - 1)] 
            except:
                previous_steps = "There is no previous steps and observations."
        else:
            previous_steps = "There is no previous steps and observations."
        # try:
        self.dpo_logger.logger.info(f"-----------------BEAM SEARCH {idx} BEFORE RANKER-----------------")
        self.dpo_logger.logger.info(f"PREVIOUS_STEP: {previous_steps}")
        self.dpo_logger.logger.info(f"OBSERVATIONS: {observations}")
        self.dpo_logger.logger.info(f"-----------------BEAM SEARCH {idx} READY TO DO RANKING-----------------")
        try:
            
            # beam_rank = self.step_verifier.forward(self.beam_size, self.task_tmp, previous_steps, observations, llm_outputs, self.image_paths,self.captions)
            beam_rank = self.step_verifier.forward(self.beam_size, self.task_tmp, previous_steps, observations, llm_outputs,images=None,captions=self.captions)
            if len(str(beam_rank)) < 10:
                beam_rank = {"rank": [0,1,2,3,4], "reason":"Qwen Ranker error, failed in llm output. Using the default rank."}
            if min(beam_rank['rank']) > 0 :
                beam_rank['rank'] = [i - 1 for i in beam_rank['rank']]
        except:    
            beam_rank = {"rank": [0,1,2,3,4], "reason":"Qwen Ranker error, failed in llm output. Using the default rank."}
        self.dpo_logger.logger.info(f"-----------------BEAM SEARCH {idx} AFTER RANKING-----------------")
        self.dpo_logger.logger.info(f"beam_rank: {beam_rank}")
        current_step_logs["rank_reason"] = beam_rank['reason']
        best_step_idx = beam_rank['rank'][0]
        current_step_logs["best_step_idx"] = beam_rank['rank'][0]
        current_step_logs["beam_rank"] = beam_rank['rank']
        try:
            self.dpo_logger.logger.info(current_step_logs[f"observation_{best_step_idx}"])
        except:
            self.dpo_logger.logger.info("No observation found for the best step.")

        ##################### Record for agent memory. ####################
        try:
            current_step_logs[f"observation"] = current_step_logs[f"observation_{best_step_idx}"]
            current_step_logs[f"llm_output"] = current_step_logs[f"llm_output_{best_step_idx}"]
            current_step_logs[f"tool_call"] = current_step_logs[f"tool_call_{best_step_idx}"]
            current_step_logs[f"rationale"] = current_step_logs[f"rationale_{best_step_idx}"]
        except:
            self.dpo_logger.logger.info("Last Step.")     


        ### Record the final answer
        # code_action = parse_code_blob(current_step_logs[f"llm_output_{best_step_idx}"])
        # code_action = parse_code_blob(current_step_logs[f"llm_output_{best_step_idx}"])
        
        
        # current_step_logs[f"tool_call_{idx}"] = {"tool_name": "code interpreter", "tool_arguments": code_action}
        # code_actions = []
        # for idx in range(self.beam_size):
        #     code_actions.append(current_step_logs[f"tool_call_{idx}"]["tool_arguments"])

        try:
            if 'Final Answer:' in observations[best_step_idx]:
                final_answer = observations[best_step_idx]
                current_step_logs["final_answer"] = final_answer
    
            # for idx, code_action in enumerate(code_actions):
            #     if idx == best_step_idx:
            #         for line in code_action.split("\n"):
            #             if line[: len("final_answer")] == "final_answer":
            #                 self.logger.warning(">>> Final answer:")
            #                 self.logger.log(32, result)
            #                 current_step_logs["final_answer"] = result
        except Exception as e:
            # if error_msg:
            #     raise AgentParsingError(error_msg)
            # else:
            error_msg = f"Error: {e}."
            raise AgentParsingError(error_msg)
        
        return current_step_logs
    
    def write_inner_memory_from_logs(self, summary_mode: Optional[bool] = False) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs into a series of messages
        that can be used as input to the LLM.
        """
        prompt_message = {"role": MessageRole.SYSTEM, "content": self.logs[0]["system_prompt"]}
        task_message = {
            "role": MessageRole.USER,
            "content": "Task: " + self.logs[0]["task"],
        }
        if summary_mode:
            memory = [task_message]
        else:
            memory = [prompt_message, task_message]
        for i, step_log in enumerate(self.logs[1:]):
            if "llm_output" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": step_log["llm_output"].strip()}
                memory.append(thought_message)
            
            if "facts" in step_log:
                thought_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": "[FACTS LIST]:\n" + step_log["facts"].strip(),
                }
                memory.append(thought_message)

            if "plan" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": "[PLAN]:\n" + step_log["plan"].strip()}
                memory.append(thought_message)

            if "tool_call" in step_log and summary_mode:
                tool_call_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": f"[STEP {i} TOOL CALL]: " + str(step_log["tool_call"]).strip(),
                }
                memory.append(tool_call_message)

            if "task" in step_log:
                tool_call_message = {
                    "role": MessageRole.USER,
                    "content": "New task:\n" + step_log["task"],
                }
                memory.append(tool_call_message)

            if "error" in step_log or "observation" in step_log:
                if "error" in step_log:
                    message_content = (
                        f"[OUTPUT OF STEP {i}] Error: "
                        + str(step_log["error"])
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                elif "observation" in step_log:
                    message_content = f"[OUTPUT OF STEP {i}] Observation:\n{step_log['observation']}"
                tool_response_message = {"role": MessageRole.TOOL_RESPONSE, "content": message_content}
                memory.append(tool_response_message)

        return memory
    
    def process_save_json(self, json_data, final_answer,gt=None):
        new_json_data = []
        for idx, item in enumerate(json_data):
            tmp = {"step number": idx}
            if idx == 0:    
                tmp["task"] = item["task"]
                tmp["image_paths"] = self.image_paths   
                if gt is not None:
                    tmp["gt"] = gt

            if "beam_size" in item:
                print(item.keys())

                tmp["beam_size"] = item["beam_size"]
                for i in range(item["beam_size"]):
                    try:
                        tmp[f"llm_output_{i}"] = item[f"llm_output_{i}"]
                    except:
                        print(f" STEP {idx}, BEAM {i} : LLM OUTPUT NOT FOUND.")
                        pass
                    try:
                        tmp[f"rationale_{i}"] = item[f"rationale_{i}"]
                        tmp[f"tool_call_{i}"] = item[f"tool_call_{i}"]
                        tmp[f"observation_{i}"] = item[f"observation_{i}"]
                    except:
                        print(f" STEP {idx}, BEAM {i} : ERROR occured.")
                        pass
            elif "error" in item:
                tmp["error"] = str(item["error"])
            try:
                tmp["beam_rank"] = str(item["beam_rank"])
                tmp["best_step_idx"] = str(item["best_step_idx"])
                tmp["rank_reason"] = str(item["rank_reason"])
            except Exception as e:
                print(f" STEP {idx} : BEAM RANK NOT FOUND. ERROR: {str(e)}")
                pass

            # if final_answer is not None:
            tmp['final_answer'] = final_answer  
            if 'final_answer' in item:
                # try:
                #     if idx == tmp["best_step_idx"]:
                #         tmp['best_final_answer'] = item['best_final_answer']
                # except:
                #     pass
                tmp['final_answer'] = item['final_answer']
                
            # tmp["llm_outputs"] = item["llm_outputs"]
            # tmp["beam_rank"] = item["beam_rank"]
            # tmp["llm_output"] = item["llm_output"]
            
            new_json_data.append(tmp)
        return new_json_data

    def save_trajectory(self, path=None, ground_truth=None, final_answer=None, gt=None) -> str:
        if path is None:
            path = os.path.join(CACHE_FOLDER+'_qwen_7b_failed', self.save_folder_id)
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        print('write to', path)
        # agent_memory = self.write_inner_memory_from_logs()
        saved_data = dict()
        # saved_data["conversations"] = agent_memory
        saved_data["logs"] = self.logs
        saved_data["final_answer"] = str(final_answer)
        saved_data["ground_truth"] = ground_truth

        beam_search_data = self.process_save_json(self.logs, final_answer,str(ground_truth))    

        try:                    
            with open(os.path.join(path, "beam_search_data.json"), "w") as f:
                json.dump(beam_search_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error: {e}")
            pass

        try:
            with open(os.path.join(path, "agent_memory.json"), "w") as f:
                json.dump(saved_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error: {e}")
            pass

        try:
            with open(os.path.join(path, "final_answer.json"), "w") as f:
                final_answer = dict(
                    final_answer = final_answer,
                    ground_truth = ground_truth,
                )
                json.dump(final_answer, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error: {e}")
            pass
            

        print(self.state)
        
        for k, v in self.state.items():
            if type(v) is not str:
                continue
            if os.path.exists(v) and not os.path.isdir(v):
                shutil.copy(v, path)
        print('save to', path)
        return path
    
from tongagent.llm_engine.mini_cpm import MiniCPMEngine
from tongagent.llm_engine import get_llm_engine
def create_agent(
        llm_engine = "tonggpt",
        task = "gta",
        error_tolerance = 3,
        lora_path = None,
        disable_vision = False,
        **kwargs
    ) -> DPOAgent:
    sample_size = kwargs.get("sample_size", 5)
    verifier = kwargs.get("verifier", 'best_selector')

    print("create_agent called", llm_engine, task, error_tolerance, lora_path, disable_vision)
    llm_engine = get_llm_engine(
        engine_type=llm_engine, 
        lora_path=lora_path,
        disable_vision=disable_vision
    )
    
    tool_boxes = []
    if task == "gta":
        tool_boxes = get_visual_model_tool_box()
    else:
        tool_boxes = get_visual_model_tool_box_for_gaia()
    tool_boxes.append(SearchTool())
    react_agent = DPOAgent(
        llm_engine=llm_engine,
        # tools=TASK_SOLVING_TOOLBOX+WEB_TOOLS,
        tools=tool_boxes,
        max_iterations=8,
        verbose=0,
        # memory_verbose=True,
        system_prompt=DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        add_base_tools=False,
        additional_authorized_imports=[
            "requests",
            "zipfile",
            "os",
            "pandas",
            "numpy",
            "sympy",
            "json",
            "bs4",
            "pubchempy",
            "xml",
            "yahoo_finance",
            "Bio",
            "sklearn",
            "scipy",
            "pydub",
            "io",
            "PIL",
            "chess",
            "PyPDF2",
            "pptx",
            "torch",
            "datetime",
            "csv",
            "fractions",
            "matplotlib",
            "pickle",
            "cv2"
        ],
        planning_interval=None,
        error_tolerance_count=error_tolerance,
        sample_size=sample_size,
        verifier=verifier,
    )
    return react_agent

