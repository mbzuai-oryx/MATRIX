"""Script to convert image content to image files using BGE model for similarity matching.

This script processes image content data, matches it with existing images using BGE model
embeddings, and copies the matched images to a target directory.
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

# Constants
QUERY_EMBEDDING_SAVE_PATH = 'data_generation/sharegpt4v/support_embedding_sharegpt4v_100k_chartqa_all.npy'
IMAGE_BASE_PATH = 'data_generation/sharegpt4v/data'
CAPTION_DATA_PATH = 'data_generation/sharegpt4v/chartqa_sharegpt4v_all.json'
FILE_SAVE_ROOT_PATH = 'data/tongagent'


def load_json(file_path: str) -> Union[Dict, List]:
    """Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Loaded JSON data as dictionary or list.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(file_path: str, data: Union[Dict, List]) -> None:
    """Save data to a JSON file.

    Args:
        file_path: Path where to save the JSON file.
        data: Data to save (dictionary or list).
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def extract_file_name(file_path: str) -> str:
    """Extract the file name from a full file path.

    Args:
        file_path: Full path to the file.

    Returns:
        The base name of the file.
    """
    return os.path.basename(file_path)


def process_image_content(
    source_path: str,
    save_path: str,
    bge_model: BGEM3FlagModel,
    support_embedding: np.ndarray,
    source_image_caption: List[Dict]
) -> None:
    """Process image content and match with existing images.

    Args:
        source_path: Path to the source JSON file containing queries and image content.
        save_path: Path where to save the processed results.
        bge_model: BGE model for encoding text.
        support_embedding: Pre-computed embeddings for the support set.
        source_image_caption: List of image captions and paths.
    """
    data = load_json(source_path)
    imagepath_list = []
    count = 0

    for task in tqdm(data, desc="Processing tasks"):
        new_task = {}
        
        # Extract query
        query = task.get("Query") or task.get("query")
        if not query:
            continue

        new_task["query"] = query
        new_task["tools"] = task["tools"]

        # Process files
        try:
            file_json = task["file"]
        except KeyError:
            print("No files found in task")
            continue

        source_paths = list(file_json["image_content"].values())
        query_image_num = len(source_paths)
        new_task["file_num"] = query_image_num

        if query_image_num == 0:
            imagepath_list.append(task)
            count += 1
            continue

        # Process image embeddings and matching
        try:
            query_image_embedding = bge_model.encode(source_paths)['dense_vecs']
        except Exception as e:
            print(f"Error in query embedding extraction: {e}")
            continue

        similarity = query_image_embedding @ support_embedding.T
        max_indices = np.argmax(similarity, axis=1)

        # Process matched images
        files = []
        file_information = {}

        for i in range(query_image_num):
            matched_image = source_image_caption[max_indices[i]]
            path = matched_image["image"]
            caption = matched_image["caption"]

            if not os.path.exists(os.path.join(IMAGE_BASE_PATH, path)):
                print(f"Image does not exist: {os.path.join(IMAGE_BASE_PATH, path)}")
                continue

            file_name = extract_file_name(path)
            file_information[f"file_{i+1}"] = {
                "file_content": list(file_json["image_content"].values())[i],
                "file_path": file_name
            }

            try:
                target_path = os.path.join(FILE_SAVE_ROOT_PATH, file_name)
                if not os.path.exists(target_path):
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy(
                        os.path.join(IMAGE_BASE_PATH, path),
                        os.path.join(FILE_SAVE_ROOT_PATH, os.path.dirname(path))
                    )
            except Exception as e:
                print(f"Image copy failed: {os.path.join(IMAGE_BASE_PATH, path)}")
                print(f"Error: {e}")

            files.append({
                'type': 'image',
                'path': path,
                'caption': caption
            })

        new_task["files"] = files
        new_task["file_information"] = file_information
        imagepath_list.append(new_task)
        count += 1

    print(f"Processed {count} tasks")
    save_json(save_path, imagepath_list)


def image_content2image_file(args: argparse.Namespace) -> None:
    """Main function to process image content to image files.

    Args:
        args: Command line arguments containing configuration.
    """
    source_path = os.path.join(args.save_path, 'file/merged_json/query2image_content.json')
    save_path = os.path.join(args.save_path, 'file/merged_json/image_content2image_file.json')

    # Load image captions
    source_image_caption = load_json(CAPTION_DATA_PATH)

    # Initialize BGE model
    bge_model = BGEM3FlagModel(
        'BAAI/bge-m3',
        use_fp16=True
    )

    # Load support embeddings
    support_embedding = np.load(QUERY_EMBEDDING_SAVE_PATH)

    # Process image content
    process_image_content(
        source_path,
        save_path,
        bge_model,
        support_embedding,
        source_image_caption
    )


def main() -> None:
    """Main entry point of the script."""
    with open('data_generation/gta_pipeline/_timestamp.txt', 'r', encoding='utf-8') as f:
        timestamp = f.read().strip()

    parser = argparse.ArgumentParser(description='Generate queries using GTA data')
    parser.add_argument("--number", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--save-path",
        type=str,
        default=f'data_generation/gta_pipeline/save/{timestamp}/'
    )
    args = parser.parse_args()

    image_content2image_file(args)


if __name__ == "__main__":
    main()