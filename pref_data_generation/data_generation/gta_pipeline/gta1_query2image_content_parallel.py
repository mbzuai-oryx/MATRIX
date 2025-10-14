"""Script for generating image content from queries using Azure OpenAI API in parallel.

This script processes queries in parallel to generate image content using Azure OpenAI's API.
It handles rate limiting, error cases, and saves the results in JSON format.
"""

import argparse
import json
import os
import random
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AzureOpenAI
from tqdm import tqdm

from merge import merge

# Configuration constants
REGION = os.getenv('REGION')
MODEL = os.getenv('MODEL')
API_KEY = os.getenv('API_KEY')
API_BASE = os.getenv('API_BASE')
ENDPOINT = f"{API_BASE}/{REGION}"
NUM_SECONDS_TO_SLEEP = 10
MAX_TOKENS = 2048
TEMPERATURE = 1

# Path constants
SYSTEM_PROMPT_PATH = 'data_generation/gta_pipeline/prompts/file/gta1_system.prompt'
USER_PROMPT_PATH = 'data_generation/gta_pipeline/prompts/file/gta1_user.prompt'

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON data from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary containing the JSON data.
    """
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Save data to a JSON file.

    Args:
        path: Path where to save the JSON file.
        data: Data to save in JSON format.
    """
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def get_chat_response(
    messages: List[Dict[str, Any]],
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    n: int = 1,
    patience: int = 10,
    sleep_time: int = 2
) -> Union[str, List[str]]:
    """Get response from Azure OpenAI API with retry mechanism.

    Args:
        messages: List of message dictionaries for the chat.
        model: Model to use for completion.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.
        n: Number of completions to generate.
        patience: Number of retry attempts.
        sleep_time: Time to sleep between retries.

    Returns:
        Generated response text or list of responses.
    """
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
            print(f"Error in chat response: {e}")
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def get_dialogue(content: str, max_tokens: int) -> str:
    """Get dialogue response from the API.

    Args:
        content: Input content for the dialogue.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        Generated dialogue response.
    """
    try:
        messages = [{
            'role': 'user',
            'content': [{
                "type": "text",
                "text": content
            }]
        }]
        return get_chat_response(
            messages=messages,
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"Error in dialogue generation: {e}")
        time.sleep(NUM_SECONDS_TO_SLEEP)
        return ""


def get_dialogue_system(
    content: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int
) -> str:
    """Get dialogue response with system prompt.

    Args:
        content: Input content for the dialogue.
        system_prompt: System prompt to use.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        Generated dialogue response.
    """
    try:
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': [{
                    "type": "text",
                    "text": content
                }]
            }
        ]
        return get_chat_response(
            messages=messages,
            model=MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"Error in system dialogue generation: {e}")
        time.sleep(NUM_SECONDS_TO_SLEEP)
        return ""


def extract_between(start_str: str, end_str: str, text: str) -> Optional[str]:
    """Extract text between two strings.

    Args:
        start_str: Starting string to look for.
        end_str: Ending string to look for.
        text: Text to search in.

    Returns:
        Extracted text or None if not found.
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
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return None


def generate_identifier(length: int = 16) -> str:
    """Generate a random identifier.

    Args:
        length: Length of the identifier.

    Returns:
        Random identifier string.
    """
    characters = string.ascii_letters + string.digits + '_-'
    return ''.join(random.choices(characters, k=length))


def query2image_content_single_json(
    item: Dict[str, Any],
    user_prompt_ori: str,
    system_prompt_ori: str,
    save_path: str
) -> None:
    """Process a single query to generate image content.

    Args:
        item: Query item containing the query text.
        user_prompt_ori: Original user prompt template.
        system_prompt_ori: Original system prompt.
        save_path: Path to save the results.
    """
    query = item.get("Query") or item.get("query")
    if not query:
        return

    output_dir = os.path.join(save_path, 'file/query2image_content_single/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path_new = os.path.join(
        output_dir,
        f"{generate_identifier()}.json"
    )
    
    user_prompt = user_prompt_ori.replace('<query>', query)
    dialogue = get_dialogue_system(
        user_prompt,
        system_prompt_ori,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    if '###' in dialogue:
        json_string = extract_between('### json start', '### json end', dialogue)
    elif '```' in dialogue:
        json_string = extract_between('```json\n', '```', dialogue)
    else:
        json_string = dialogue

    if json_string is not None:
        try:
            file_json = json.loads(json_string)
            item['file'] = file_json['file']
            save_json(save_path_new, item)
        except json.JSONDecodeError as e:
            print(f"Error in JSON saving: {e}")
    else:
        print('JSON is None')


def multi_process_query2image_content(args: argparse.Namespace) -> None:
    """Process multiple queries in parallel to generate image content.

    Args:
        args: Command line arguments.
    """
    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as file:
        system_prompt_ori = file.read()

    with open(USER_PROMPT_PATH, 'r', encoding='utf-8') as file:
        user_prompt_ori = file.read()

    source_dir = os.path.join(args.save_path, 'query/queries_merged/')
    source_file = next(f for f in os.listdir(source_dir) if f.endswith('.json'))
    source_path = os.path.join(source_dir, source_file)

    output_dir = os.path.join(args.save_path, 'file')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = load_json(source_path)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for item in tqdm(data):
            executor.submit(
                query2image_content_single_json,
                item,
                user_prompt_ori,
                system_prompt_ori,
                args.save_path
            )

    print(f'Query to image content of {len(data)} samples cost time: '
          f'{time.time() - start_time:.2f} s')


def main() -> None:
    """Main function to run the script."""
    with open('data_generation/gta_pipeline/_timestamp.txt', 'r', encoding='utf-8') as f:
        timestamp = f.read().strip()

    parser = argparse.ArgumentParser(description='Generate queries using GTA data')
    log_dir = f'data_generation/gta_pipeline/log/{timestamp}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'{timestamp}_1_q2content_parallel.log')
    sys.stdout = open(log_file, 'a', encoding='utf-8')

    parser.add_argument("--number", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--save-path",
        type=str,
        default=f'data_generation/gta_pipeline/save/{timestamp}/'
    )
    args = parser.parse_args()

    multi_process_query2image_content(args)
    merge(
        os.path.join(args.save_path, 'file/query2image_content_single/'),
        os.path.join(args.save_path, 'file/merged_json/'),
        'query2image_content.json'
    )


if __name__ == '__main__':
    main()