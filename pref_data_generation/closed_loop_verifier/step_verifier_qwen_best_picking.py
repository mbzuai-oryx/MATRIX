"""Module for step verification using Qwen models."""

from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


class QwenModel:
    """A class for handling Qwen model operations.

    This class provides functionality for both standard Qwen models and Qwen-VL models.
    It handles model initialization, text extraction, and forward passes.

    Attributes:
        model_name: Name of the model to use.
        lora_path: Optional path to LoRA weights.
        model: The loaded model instance.
        processor: Processor for VL models.
        tokenizer: Tokenizer for standard models.
        system_prompt_ori: Original system prompt.
        user_prompt_ori: Original user prompt.
    """

    def __init__(self, model_name: str, lora_path: Optional[str] = None) -> None:
        """Initialize the Qwen model.

        Args:
            model_name: Name of the model to use.
            lora_path: Optional path to LoRA weights.
        """
        self.model_name = model_name
        self.lora_path = lora_path

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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        with open('closed_loop_verifier/prompt/best_verifier_system.prompt', 'r',
                 encoding='utf-8') as file:
            self.system_prompt_ori = file.read()

        with open('closed_loop_verifier/prompt/best_verifier_user.prompt', 'r',
                 encoding='utf-8') as file:
            self.user_prompt_ori = file.read()

    def extract_between(self, start_str: str, end_str: str, text: str) -> Optional[str]:
        """Extract text between two strings.

        Args:
            start_str: The starting string to search for.
            end_str: The ending string to search for.
            text: The text to search in.

        Returns:
            The extracted text between start_str and end_str, or None if not found.
        """
        try:
            start_index = text.find(start_str)
            if start_index == -1:
                return None

            start_index += len(start_str)
            end_index = text.find(end_str, start_index)
            if end_index == -1:
                return None

            return text[start_index:end_index]
        except Exception:
            return None

    def forward(self, messages: List[Dict[str, Any]]) -> str:
        """Perform a forward pass through the model.

        Args:
            messages: List of message dictionaries containing the conversation.

        Returns:
            The model's response as a string.
        """
        if "VL" in self.model_name:
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
            inputs = inputs.to("cuda")

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                top_k=100,
                do_sample=True,
                repetition_penalty=1.05,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(
                    inputs.input_ids, generated_ids
                )
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(
                    model_inputs.input_ids, generated_ids
                )
            ]

            output_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return output_text


class VLLMQwenModel(QwenModel):
    """A class for handling Qwen models using VLLM.

    This class extends QwenModel to use VLLM for inference.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the VLLM Qwen model.

        Args:
            model_name: Name of the model to use.
        """
        super().__init__(model_name)
        from tongagent.utils import load_config
        from openai import OpenAI

        config = load_config()
        self.endpoint = config.verifier.endpoint
        openai_api_key = "EMPTY"
        self.openai_api_base = f"http://{self.endpoint}:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=self.openai_api_base,
        )

    def forward(self, messages: List[Dict[str, Any]]) -> str:
        """Perform a forward pass using VLLM.

        Args:
            messages: List of message dictionaries containing the conversation.

        Returns:
            The model's response as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content


def write_json(data: Dict[str, Any], filename: str) -> None:
    """Write a JSON-compatible Python dictionary to a file.

    Args:
        data: The JSON-compatible dictionary to write.
        filename: The name of the file to write to.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as error:
        print(f"An error occurred while writing to the file: {error}")


class StepVerifier:
    """A class for verifying steps in a process.

    This class handles the verification of steps using Qwen models,
    supporting both standard and VLLM implementations.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        """Initialize the step verifier.

        Args:
            model: Name of the model to use for verification.
        """
        self.model = model
        from tongagent.utils import load_config
        self.config = load_config()

        if self.config.verifier.use_vllm:
            self.qwen = VLLMQwenModel("Qwen/Qwen2.5-7B-Instruct")
        else:
            if "VL" in self.model:
                self.qwen = QwenModel("Qwen/Qwen2-VL-7B-Instruct")
            else:
                self.qwen = QwenModel("Qwen/Qwen2.5-7B-Instruct")

        with open('closed_loop_verifier/prompt/best_verifier_system.prompt', 'r',
                 encoding='utf-8') as file:
            self.system_prompt_ori = file.read()

        with open('closed_loop_verifier/prompt/best_verifier_user.prompt', 'r',
                 encoding='utf-8') as file:
            self.user_prompt_ori = file.read()

    def extract_between(self, start_str: str, end_str: str, text: str) -> Optional[str]:
        """Extract text between two strings.

        Args:
            start_str: The starting string to search for.
            end_str: The ending string to search for.
            text: The text to search in.

        Returns:
            The extracted text between start_str and end_str, or None if not found.
        """
        try:
            start_index = text.find(start_str)
            if start_index == -1:
                return None

            start_index += len(start_str)
            end_index = text.find(end_str, start_index)
            if end_index == -1:
                return None

            return text[start_index:end_index]
        except Exception:
            return None

    def get_response(self, messages: List[Dict[str, Any]]) -> Union[Dict[str, Any], str]:
        """Get and process the model's response.

        Args:
            messages: List of message dictionaries containing the conversation.

        Returns:
            The processed response as a dictionary or 'error' if processing failed.
        """
        response = self.qwen.forward(messages)
        analysis_response = self.extract_between('```json\n', '\n```', response)

        try:
            analysis = json.loads(analysis_response)
        except Exception as error:
            print(f"An error occurred in the json_loads of verifier: {error}")
            return 'error'

        return analysis

    def forward(
        self,
        num_steps: int,
        task: str,
        previous_steps: List[str],
        current_observations: List[str],
        current_steps: List[str],
        images: Optional[List[str]] = None,
        captions: Optional[List[str]] = None
    ) -> Union[Dict[str, Any], str]:
        """Perform a forward pass for step verification.

        Args:
            num_steps: Number of steps to verify.
            task: The task description.
            previous_steps: List of previous steps.
            current_observations: List of current observations.
            current_steps: List of current steps.
            images: Optional list of image paths.
            captions: Optional list of image captions.

        Returns:
            The verification result as a dictionary or 'error' if verification failed.
        """
        system_prompt = self.system_prompt_ori.replace('<N>', str(num_steps))
        
        if captions:
            task += '\n' + "For your better understanding of the file content, we provide the file descriptions for the files are as follows:"
            for idx, caption in enumerate(captions):
                task += f'\nThis is the description for the files{idx+1}: {caption}\n'

        usr_prompt = self.user_prompt_ori.replace('<task>', task)

        eval_set = {}
        for i in range(num_steps):
            eval_set["PREVIOUS_RESULT"] = previous_steps
            eval_set[f'Trajectory{i+1}'] = {
                "CURRENT_STEP": current_steps[i],
                "CURRENT_RESULT": current_observations[i]
            }
        eval_set = json.dumps(eval_set, indent=4)
        usr_prompt = usr_prompt.replace('<step_set>', eval_set)

        if images and "VL" in self.model:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": image} for image in images
                ] + [{"type": "text", "text": usr_prompt}]}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_prompt}
            ]

        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        score = self.get_response(messages)

        messages.append({"role": "verifier", "content": score})
        try:
            write_json(
                messages,
                f'/home/lipengxiang/codes/DPOagent/TongAgent/closed_loop_verifier/plauground/verifier-messages-{current_time}.json'
            )
        except Exception as error:
            print(f"An error occurred while writing to the file: {error}")

        return score


def main() -> None:
    """Main function for testing the step verifier."""
    verifier = StepVerifier("Qwen/Qwen2.5-7B-Instruct")
    previous_steps = ["no previous step is none" * 3]
    current_steps = [
        "Thought: First, I need to find out the price of the NVIDIA GeForce RTX 4070 SUPER. I will use the `visualizer` tool to extract the price information from the NVIDIA's official website.\nCode:\n```py\nprice_info = visualizer(question=\"What is the price of the NVIDIA GeForce RTX 4070 SUPER?\", image_path=\"gta_dataset/image/image_14.jpg\")\nprint(price_info)\n```",
        "Thought: I will use the `visualizer` tool to get the price of the NVIDIA GeForce RTX 4070 SUPER from NVIDIA's official website. Then I will calculate the total cost by multiplying the price by the number of GPUs. Finally, I will use the `final_answer` tool to provide the final answer.\nCode:\n```py\nprice = visualizer(question=\"How much does the NVIDIA GeForce RTX 4070 SUPER cost?\", image_path=\"gta_dataset/image/image_14.jpg\")\nprint(price)\n```",
        "Thought: I need to find the price of one NVIDIA GeForce RTX 4070 SUPER on NVIDIA's official website in January. I will use the `ask_search_agent` tool to search for this information.\nCode:\n```py\n# Search for the price of the NVIDIA GeForce RTX 4070 SUPER on NVIDIA's official website in January\nsearch_result = ask_search_agent(query=\"NVIDIA GeForce RTX 4070 SUPER price January\")\nprint(search_result)\n```"
    ]

    current_observations = [
        'The NVIDIA GeForce RTX 4070 SUPER is priced at $299.',
        'The NVIDIA GeForce RTX 4070 SUPER is priced at $399.',
        'The NVIDIA GeForce RTX 4070 SUPER is priced at $699.',
        'The NVIDIA GeForce RTX 4070 SUPER costs $699.',
        'The NVIDIA GeForce RTX 4070 SUPER is priced at $599.',
        'The NVIDIA GeForce RTX 4070 SUPER is priced at $399.',
        "Code execution failed due to the following error:\nConnection error."
    ]

    current_observations = [current_observations[1], current_observations[0], current_observations[2]]
    current_steps = [current_steps[1], current_steps[0], current_steps[2]]
    task = "The men in the picture want to buy one NVIDIA GeForce RTX 4070 SUPER each. According to NVIDIA's official website in January, how many dollars will they need to spend in total?\nAttachement: gta_dataset/image/image_14.jpg"
    verifier.forward(3, task, previous_steps, current_observations, current_steps, captions=["A photo of 4070"])


if __name__ == "__main__":
    main()
