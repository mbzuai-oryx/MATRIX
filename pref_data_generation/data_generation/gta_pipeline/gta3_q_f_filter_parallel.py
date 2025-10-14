"""Script for processing and filtering GTA data using Azure OpenAI API.

This script provides functionality for processing GTA data, including image encoding,
dialogue generation, and file verification using Azure OpenAI's API.
"""

import argparse
import base64
import json
import os
import random
import string
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AzureOpenAI
from tqdm import tqdm

from merge import merge

# Constants
REGION = os.getenv('REGION')
MODEL = os.getenv('MODEL')
API_KEY = os.getenv('API_KEY')
API_BASE = os.getenv('API_BASE')
ENDPOINT = f"{API_BASE}/{REGION}"
NUM_SECONDS_TO_SLEEP = 10
MAX_TOKENS = 2048
TEMPERATURE = 1

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
)


def read_json(path: str) -> Dict[str, Any]:
    """Read JSON data from a file.

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


def encode_image(image_path: str) -> str:
    """Encode an image file to base64.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_chat_response(
    messages: List[Dict[str, Any]],
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    n: int = 1,
    patience: int = 10,
    sleep_time: int = 2
) -> Union[str, List[str]]:
    """Get response from Azure OpenAI chat completion API.

    Args:
        messages: List of message dictionaries.
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
            print(f"Error in get_chat_response: {e}")
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def get_dialogue(content: str, max_tokens: int) -> str:
    """Get dialogue response for given content.

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
        response = get_chat_response(
            messages=messages,
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"Error in get_dialogue: {e}")
        time.sleep(NUM_SECONDS_TO_SLEEP)
        return ""

    return response


def get_dialogue_system(
    content: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int
) -> str:
    """Get dialogue response with system prompt.

    Args:
        content: Input content for the dialogue.
        system_prompt: System prompt to guide the response.
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
        response = get_chat_response(
            messages=messages,
            model=MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"Error in get_dialogue_system: {e}")
        time.sleep(NUM_SECONDS_TO_SLEEP)
        return ""

    return response


def get_dialogue_system_oneimage(
    content: str,
    system_prompt: str,
    images: List[str],
    temperature: float,
    max_tokens: int = MAX_TOKENS
) -> str:
    """Get dialogue response with system prompt and one image.

    Args:
        content: Input content for the dialogue.
        system_prompt: System prompt to guide the response.
        images: List of image paths.
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
        base64_images = [encode_image(image_path) for image_path in images]

        for base64_image in base64_images:
            messages[1]['content'].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        response = get_chat_response(
            messages=messages,
            model=MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"Error in get_dialogue_system_oneimage: {e}")
        time.sleep(NUM_SECONDS_TO_SLEEP)
        return ""

    return response


def get_dialogue_system_twoimage(
    content: str,
    system_prompt: str,
    image_path_1: str,
    image_path_2: str,
    temperature: float,
    max_tokens: int = MAX_TOKENS
) -> str:
    """Get dialogue response with system prompt and two images.

    Args:
        content: Input content for the dialogue.
        system_prompt: System prompt to guide the response.
        image_path_1: Path to first image.
        image_path_2: Path to second image.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        Generated dialogue response.
    """
    try:
        base64_image_1 = encode_image(image_path_1)
        base64_image_2 = encode_image(image_path_2)

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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image_2}"}
                    }
                ]
            }
        ]
        response = get_chat_response(
            messages=messages,
            model=MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"Error in get_dialogue_system_twoimage: {e}")
        time.sleep(NUM_SECONDS_TO_SLEEP)
        return ""

    return response


def extract_between(str1: str, str2: str, str3: str) -> Optional[str]:
    """Extract substring between two strings.

    Args:
        str1: First string to find.
        str2: Second string to find.
        str3: String to search in.

    Returns:
        Extracted substring or None if not found.
    """
    try:
        start_index = str3.find(str1)
        if start_index == -1:
            return None

        start_index += len(str1)
        end_index = str3.find(str2, start_index)
        if end_index == -1:
            return None

        return str3[start_index:end_index]
    except Exception as e:
        print(f"Error in extract_between: {e}")
        return None


def traj_extraction(traj_json: List[Dict[str, str]]) -> str:
    """Extract trajectory from JSON data.

    Args:
        traj_json: List of trajectory steps.

    Returns:
        Concatenated trajectory string.
    """
    trajectory = ''
    for step in traj_json:
        if step['role'] == 'assistant':
            trajectory_step = step['content']
            trajectory = trajectory + trajectory_step + '\n'
        elif step['role'] == 'user':
            trajectory_step = step['content']
            trajectory = trajectory + trajectory_step + '\n'
    return trajectory


def fetch_content(item: Dict[str, Any]) -> List[str]:
    """Fetch content from file information.

    Args:
        item: Dictionary containing file information.

    Returns:
        List of file contents.
    """
    if "file_information" not in item:
        return []
    file = item["file_information"]
    content = []
    for i in range(len(file)):
        j = file[f'file_{str(i+1)}']
        content.append(j['file_content'])
    return content


def generate_identifier(length: int = 16) -> str:
    """Generate random identifier.

    Args:
        length: Length of the identifier.

    Returns:
        Randomly generated identifier string.
    """
    characters = string.ascii_letters + string.digits + '_-'
    return ''.join(random.choices(characters, k=length))


def file_verifier_single(
    item: Dict[str, Any],
    user_prompt_ori: str,
    system_prompt_ori: str,
    image_path: str,
    file_filtered_folder: str
) -> None:
    """Verify single file and save results.

    Args:
        item: File information to verify.
        user_prompt_ori: Original user prompt.
        system_prompt_ori: Original system prompt.
        image_path: Path to images.
        file_filtered_folder: Folder to save filtered files.
    """
    save_path = os.path.join(file_filtered_folder, generate_identifier() + '.json')
    
    if 'correct' in item.keys():
        if item['correct'] == 'yes':
            save_json(save_path, item)
            return
        query = item['updated_query']
    else:
        query = item['query']

    images = []
    caption = []
    if 'files' in item:
        for file in item['files']:
            if file['type'] == 'image':
                try:
                    images.append(os.path.join(image_path, file["path"]))
                    caption.append(file['caption'])
                except FileNotFoundError:
                    print(f"Image not found: {file['path']}")

    user_prompt = user_prompt_ori.replace('<query>', query)

    dialogue = get_dialogue_system_oneimage(
        user_prompt,
        system_prompt_ori,
        images,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    print(f"Dialogue: {dialogue}")
    
    try:
        dialogue_json = dialogue.split('```json')[1].split('```')[0]
        dialogue_json = json.loads(dialogue_json)
    except:
        try:
            dialogue_json = dialogue.split('### start json')[1].split('### end json')[0]
            dialogue_json = json.loads(dialogue_json)
        except:
            dialogue_json = dialogue
            print(f"Dialogue format error, failure in json save: {dialogue}")
            return

    item['correct'] = dialogue_json['correct']
    item['updated_query'] = dialogue_json['updated_query']
    save_json(save_path, item)


def file_verifier(args: argparse.Namespace) -> None:
    """Main file verification process.

    Args:
        args: Command line arguments.
    """
    # Prompt setting
    system_prompt_path = 'data_generation/gta_pipeline/prompts/file/gta3_file_verifier_system.prompt'
    user_prompt_path = 'data_generation/gta_pipeline/prompts/file/gta3_file_verifier_user.prompt'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    with open(system_prompt_path, 'r', encoding='utf-8') as file:
        system_prompt_ori = file.read()

    with open(user_prompt_path, 'r', encoding='utf-8') as file:
        user_prompt_ori = file.read()

    # Image base path
    image_path = 'data/tongagent'

    iteration = [
        {
            "source_path": f"{args.save_path}/file/merged_json/image_content2image_file.json",
            "filtered_save_folder": f"{args.save_path}/file/file_filtered_v1/",
            "filtered_save_path": f"{args.save_path}/file/merged_json/",
            "filtered_save_name": "image_file_filtered_v1.json"
        },
        {
            "source_path": f"{args.save_path}/file/merged_json/image_file_filtered_v1.json",
            "filtered_save_folder": f"{args.save_path}/file/file_filtered_v2/",
            "filtered_save_path": f"{args.save_path}/file/merged_json/",
            "filtered_save_name": "image_file_filtered_v2.json"
        }
    ]

    for idx, iter_data in enumerate(iteration):
        source_path = iter_data["source_path"]
        filtered_save_path = iter_data["filtered_save_path"]
        filtered_save_name = iter_data["filtered_save_name"]
        file_filtered_folder = iter_data["filtered_save_folder"]

        if not os.path.exists(file_filtered_folder):
            os.makedirs(file_filtered_folder)

        data = read_json(source_path)

        start = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for item in tqdm(data):
                executor.submit(
                    file_verifier_single,
                    item,
                    user_prompt_ori,
                    system_prompt_ori,
                    image_path,
                    file_filtered_folder
                )

        print(f'File verifier {idx} of {len(data)} samples cost time: {time.time() - start} s')
        merge(file_filtered_folder, filtered_save_path, filtered_save_name)

        new_data = read_json(os.path.join(filtered_save_path, filtered_save_name))
        correctness = [item['correct'] for item in new_data]
        counter = Counter(correctness)
        print(f"Count of yes: {counter['yes']}")
        print(f"Count of no: {counter['no']}")


def main() -> None:
    """Main entry point of the script."""
    with open('data_generation/gta_pipeline/_timestamp.txt', 'r', encoding='utf-8') as f:
        timestamp = f.read().strip()

    parser = argparse.ArgumentParser(description='Generate queries using GTA data')
    
    log_dir = f'data_generation/gta_pipeline/log/{timestamp}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    sys.stdout = open(f'{log_dir}/{timestamp}_3_verifier.log', 'a')
    
    parser.add_argument("--number", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--save-path", type=str, default=f'data_generation/gta_pipeline/save/{timestamp}/')
    
    args = parser.parse_args()
    file_verifier(args)


if __name__ == "__main__":
    main()
