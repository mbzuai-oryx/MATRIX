"""
GTA Query Generation Module

This module handles the generation of queries using GPT models through Azure OpenAI.
It includes functions for processing JSON data, generating prompts, and managing API calls.
"""

import argparse
import json
import os
import random
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

import openai
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
REGION = os.getenv('REGION')
MODEL = os.getenv('MODEL')
API_KEY = os.getenv('API_KEY')
API_BASE = os.getenv('API_BASE')
ENDPOINT = f"{API_BASE}/{REGION}"

# Constants
NUM_SECONDS_TO_SLEEP = 3
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# Tool mapping configuration
TOOL_MAP = {
    "ImageDescription": "visualizer",
    "RegionAttributeDescription": "visualizer",
    "CountGivenObject": "visualizer",
    "TextToImage": "image_generator",
    "ImageStylization": "image_edit",
    "GoogleSearch": "WebSearchTool",
    "TextToBbox": "ask_search_agent",
    "MathOCR": "visualizer",
    "OCR": "visualizer",
    "DrawBox": "PythonInterpreter",
    "Solver": "PythonInterpreter",
    "Calculator": "PythonInterpreter",
    "AddText": "PythonInterpreter",
    "Plot": "PythonInterpreter"
}

def setup_logging(timestamp: str) -> None:
    """Set up logging directory and redirect stdout to log file."""
    os.makedirs(f'data_generation/gta_pipeline/log/{timestamp}/', exist_ok=True)
    sys.stdout = open(f'data_generation/gta_pipeline/log/{timestamp}/{timestamp}_0_query.log', 'a')

def merge(path: str, timestamp: str) -> None:
    """Merge multiple JSON files into a single file.
    
    Args:
        path: Base path for JSON files
        timestamp: Timestamp for the merged file
    """
    json_path = os.path.join(path, 'query/query_json/')
    merge_save_path = os.path.join(path, 'query/queries_merged')
    
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    data = []
    
    for json_file in json_files:
        with open(os.path.join(json_path, json_file), encoding='utf-8') as file:
            tmp = json.load(file)
            if isinstance(tmp, list) and len(tmp) == 1:
                tmp = tmp[0]
            data.extend(tmp)
    
    length = len(data)
    os.makedirs(merge_save_path, exist_ok=True)
    
    with open(f"{merge_save_path}/gta_query_num{length}_{timestamp}.json", 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    print(f"Successfully merged {length} json files")

def mapping(tools: List[str]) -> List[str]:
    """Map tool names to their corresponding categories.
    
    Args:
        tools: List of tool names to map
        
    Returns:
        List of mapped tool categories
    """
    mapped_tools = [TOOL_MAP[t] for t in tools]
    return list(set(mapped_tools))

def extract_user_content_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract user content and tools from JSON data.
    
    Args:
        data: JSON data containing user content and tools
        
    Returns:
        List of dictionaries containing queries and tools
    """
    user_contents = []
    for i in range(len(data)):
        case_data = data[str(i)]
        tmp = {'query': "", "tools": []}
        tools = case_data.get('tools', [])
        
        for tool in tools:
            tmp['tools'].append(tool['name'])
            
        dialogs = case_data.get('dialogs', [])
        for dialog in dialogs:
            if dialog.get('role') == 'user':
                tmp["query"] = dialog.get('content', '') + '\n'
                
        tmp['tools'] = mapping(tmp['tools'])
        print(f"----------------------------------{tmp['tools']}")
        user_contents.append(tmp)

    return user_contents

def sample_in_context_examples(pool: List[Dict[str, Any]], num: int = 10) -> List[Dict[str, Any]]:
    """Randomly sample examples from a pool.
    
    Args:
        pool: List of examples to sample from
        num: Number of examples to sample
        
    Returns:
        List of sampled examples
    """
    return random.sample(pool, num)

def prompt_with_random_examples(json_data: Dict[str, Any], prompt: str, num: int = 10) -> str:
    """Format prompt with random in-context examples.
    
    Args:
        json_data: JSON data containing examples
        prompt: Base prompt template
        num: Number of examples to include
        
    Returns:
        Formatted prompt string
    """
    user_contents = extract_user_content_from_json(json_data)
    in_context_examples = sample_in_context_examples(user_contents, num=num)
    in_context_examples = json.dumps(in_context_examples, indent=4)
    return prompt.replace('IN_CONTEXT_EXAMPLES', "".join(in_context_examples))

def load_json_prompt(json_prompt_paths: List[Dict[str, str]]) -> List[Tuple[Dict[str, Any], str]]:
    """Load JSON and prompt files from specified paths.
    
    Args:
        json_prompt_paths: List of dictionaries containing JSON and prompt file paths
        
    Returns:
        List of tuples containing JSON data and prompts
    """
    json_prompt = []
    for json_prompt_path in json_prompt_paths:
        with open(json_prompt_path['prompt'], 'r', encoding='utf-8') as file:
            prompt = file.read()
        with open(json_prompt_path['json'], 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        json_prompt.append((json_data, prompt))
    return json_prompt

def fetch_system_prompts(mode: int = 0, num: int = 10, num_incontext_examples: int = 10) -> List[str]:
    """Generate system prompts with random in-context examples.
    
    Args:
        mode: Mode for prompt generation
        num: Number of prompts to generate
        num_incontext_examples: Number of examples to include in each prompt
        
    Returns:
        List of system prompts
    """
    json_prompt = load_json_prompt([
        {
            "json": "data_generation/query_generation/GTA/source/gta_metadata.json",
            "prompt": "data_generation/query_generation/GTA/source/gta_query_geenration.prompt"
        }
    ])
    system_prompts = []
    json_data, prompt = json_prompt[mode]
    for _ in range(num):
        system_prompt = prompt_with_random_examples(json_data, prompt, num_incontext_examples)
        system_prompts.append(system_prompt)
    return system_prompts

def generate_identifier(length: int = 16) -> str:
    """Generate a random identifier string.
    
    Args:
        length: Length of the identifier
        
    Returns:
        Random identifier string
    """
    characters = string.ascii_letters + string.digits + '_-!@#$%^&*'
    return ''.join(random.choices(characters, k=length))

def queries_to_json_and_save(queries: str, save_path: str) -> str:
    """Convert queries to JSON and save to file.
    
    Args:
        queries: String containing queries
        save_path: Path to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    json_list = queries[7:-3]
    print(json_list)
    json_path = os.path.join(save_path, 'query/query_json/')
    json_list = json_list.replace('Query', 'query')
    json_list = json_list.replace('Tools', 'tools')
    json_list = json_list.replace('FileName', 'filename')
    
    os.makedirs(save_path, exist_ok=True)
    json_string = f'[{json_list}]'

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return ""

    output_file = os.path.join(json_path, generate_identifier() + ".json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

    print(f"Data successfully saved to {output_file}")
    return output_file

def get_chat_response(messages: List[Dict[str, Any]], model: str = MODEL,
                     temperature: float = 0.2, max_tokens: int = 2048,
                     n: int = 1, patience: int = 10, sleep_time: int = 2) -> str:
    """Get response from Azure OpenAI API.
    
    Args:
        messages: List of message dictionaries
        model: Model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        n: Number of completions
        patience: Number of retry attempts
        sleep_time: Time to sleep between retries
        
    Returns:
        Generated response text
    """
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-02-01",
        azure_endpoint=ENDPOINT,
    )
    
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0]:
                    return prediction
        except Exception as e:
            print(e)
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""

def get_queries(content: str, max_tokens: int, system_prompt: str) -> str:
    """Get queries from GPT model.
    
    Args:
        content: User prompt content
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt for the model
        
    Returns:
        Generated queries
    """
    try:
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': [
                    {
                        "type": "text",
                        "text": content
                    },
                ]
            }
        ]
        response = get_chat_response(messages=messages, model=MODEL,
                                   temperature=1, max_tokens=max_tokens)
        return response
    except openai.error.RateLimitError:
        print('openai.error.RateLimitError:')
        return ""
    except Exception as e:
        print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        return ""

def multi_process_fetch_gpt_response(user_prompt: str, system_prompt_incontext_example: str,
                                   save_path: str) -> Optional[int]:
    """Fetch GPT response using multiple processes.
    
    Args:
        user_prompt: User prompt
        system_prompt_incontext_example: System prompt with examples
        save_path: Path to save results
        
    Returns:
        1 if successful, None if failed
    """
    try:
        response = get_queries(content=user_prompt, max_tokens=2048,
                             system_prompt=system_prompt_incontext_example)
        queries_to_json_and_save(response, save_path)
        return 1
    except Exception as e:
        print(e)
        return None

def query_generation(args: argparse.Namespace) -> None:
    """Main function for query generation.
    
    Args:
        args: Command line arguments
    """
    user_prompt = "Please generate NUM_QUERIES queries. DO NOT output an id number before each query."
    user_prompt = user_prompt.replace('NUM_QUERIES', str(args.np))
    
    system_prompts = fetch_system_prompts(num=args.ngpt,
                                        num_incontext_examples=args.ni,
                                        mode=args.mode)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for system_prompt in tqdm(system_prompts):
            print(system_prompt)
            executor.submit(multi_process_fetch_gpt_response,
                          user_prompt, system_prompt, args.save_path)

    print(f"Query generation completed for {args.ngpt * args.np} queries")

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate queries using GTA data')
    parser.add_argument("--number", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--save-path", type=str,
                       default=f'data_generation/gta_pipeline/save/{TIMESTAMP}/')
    parser.add_argument("--mode", type=int, default=0,
                       help="0: mutimodal files")
    parser.add_argument('--output_path', type=str,
                       help='Path to save the generated queries',
                       default='query_generation/GTA/gta_queries.txt')
    parser.add_argument('--ngpt', '--num_gpt_queries', type=int,
                       help='Number of queries to generate', default=2)
    parser.add_argument('--ni', '--num_incontext_examples', type=int,
                       help='Number of in-context examples to include in the prompt',
                       default=20)
    parser.add_argument('--np', '--query_num_per_gpt_call', type=int,
                       help='Number of queries to generate per GPT call', default=10)
    
    args = parser.parse_args()
    print("GTA based Query GENERATION STARTED:")
    query_generation(args)
    merge(args.save_path, TIMESTAMP)

if __name__ == "__main__":
    main()