"""Module for merging JSON files from a source directory into a single output file."""

import os
import json
from typing import List, Union, Dict, Any


def merge(source_folder: str, output_folder: str, filename: str) -> None:
    """Merge multiple JSON files from source folder into a single output file.

    Args:
        source_folder: Path to the directory containing JSON files to merge.
        output_folder: Path to the directory where the merged file will be saved.
        filename: Name of the output file.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    save_path = os.path.join(output_folder, filename)
    json_files = [
        pos_json for pos_json in os.listdir(source_folder)
        if pos_json.endswith('.json')
    ]

    data: List[Union[Dict[str, Any], List[Any]]] = []
    for json_file in json_files:
        file_path = os.path.join(source_folder, json_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            tmp = json.load(file)
            if isinstance(tmp, list) and len(tmp) == 1:
                tmp = tmp[0]
            if isinstance(tmp, list):
                data.extend(tmp)
            else:
                data.append(tmp)

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

    print(f"Successfully merged {len(data)} JSON files")