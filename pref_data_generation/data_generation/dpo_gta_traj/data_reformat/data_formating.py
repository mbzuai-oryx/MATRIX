"""Data formatting module for DPO GTA trajectory data.

This module handles the formatting and processing of trajectory data for DPO training.
It includes functions for loading, processing, and saving trajectory data in a specific format.
"""

import argparse
import json
import logging
import os
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Format DPO GTA trajectory data')
    
    parser.add_argument(
        '--blue_threshold',
        type=float,
        default=0.3,
        help='BLEU score threshold for rejecting similar outputs'
    )
    
    parser.add_argument(
        '--image_path_prefix',
        type=str,
        default='data/data/tongagent/',
        help='Prefix for image paths'
    )
    
    parser.add_argument(
        '--traj_folder',
        type=str,
        default='data_generation/dpo_gta_traj/save',
        help='Folder containing trajectory data'
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        default='data_generation/dpo_gta_traj/dpo_llamafactory_format/',
        help='Path to save formatted data'
    )
    
    parser.add_argument(
        '--prompt_path',
        type=str,
        default='data_generation/dpo_gta_traj/data_reformat/system_prompt.prompt',
        help='Path to system prompt file'
    )
    
    return parser.parse_args()

def load_cases(cache_folder: str) -> List[Dict[str, Any]]:
    """Load trajectory cases from the cache folder.
    
    Args:
        cache_folder: Path to the folder containing trajectory data.
        
    Returns:
        List of loaded trajectory cases.
    """
    cases = []
    for subfolder in os.listdir(cache_folder):
        subfolder_path = os.path.join(cache_folder, subfolder)
        if os.path.isdir(subfolder_path):
            json_file_path = os.path.join(subfolder_path, "beam_search_data.json")
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, "r", encoding='utf-8') as file:
                        case_data = json.load(file)
                        cases.append(case_data)
                except json.JSONDecodeError:
                    logger.error("Error decoding JSON in %s", json_file_path)
    return cases

def generate_conversations(case: List[Dict[str, Any]], step: int, images: List[str]) -> List[Dict[str, str]]:
    """Generate conversation format from trajectory data.
    
    Args:
        case: List of trajectory steps.
        step: Current step number.
        images: List of image paths.
        
    Returns:
        List of conversation turns in the required format.
    """
    conversations_list = []
    image_len = len(images)
    task = '<image>' * image_len + "\nTask: " + case[0]['task']
    conversations_list.append({
        "from": "human",
        "value": task
    })
    
    for i in range(1, step):
        try:
            best_step_idx = case[i]['best_step_idx']
            from_gpt = case[i][f'llm_output_{best_step_idx}'] + '<end_action>'
            from_human = f"[OUTPUT OF STEP {str(i-1)}] Observation:\n" + case[i][f'observation_{best_step_idx}']
            
            conversations_list.extend([
                {"from": "gpt", "value": from_gpt},
                {"from": "human", "value": from_human}
            ])
        except KeyError:
            logger.warning("Error in case step %d, skipping", i)
            continue
            
    return conversations_list

def get_error_case_reject_indices(observations: List[str], llm_outputs: List[str]) -> List[int]:
    """Get indices of cases to reject based on format and error criteria.
    
    Args:
        observations: List of observation strings.
        llm_outputs: List of LLM output strings.
        
    Returns:
        List of indices to reject.
    """
    format_reject_list = []
    for i, output in enumerate(llm_outputs):
        output_lower = output.lower()
        if 'thought:' in output_lower and 'code' not in output_lower:
            format_reject_list.append(i)
        if 'thought:' not in output_lower:
            format_reject_list.append(i)
            
    error_reject_list = [
        i for i, obs in enumerate(observations)
        if 'error' in obs.lower()
    ]
    
    return list(set(format_reject_list + error_reject_list))

def get_bleu_selected_indices(
    chosen: str,
    rejected: List[str],
    threshold: float
) -> List[int]:
    """Get indices of rejected outputs based on BLEU score threshold.
    
    Args:
        chosen: The chosen output string.
        rejected: List of rejected output strings.
        threshold: BLEU score threshold.
        
    Returns:
        List of indices with BLEU scores below threshold.
    """
    smooth_fn = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([chosen], rej, smoothing_function=smooth_fn)
        for rej in rejected
    ]
    
    rejected_indices = [
        i for i, score in enumerate(bleu_scores)
        if score < threshold
    ]
    
    for i, score in enumerate(bleu_scores):
        if score < threshold:
            logger.debug("Rejected BLEU score: %f", score)
            
    return rejected_indices

def generate_random_id(length: int = 8) -> str:
    """Generate a random alphanumeric ID.
    
    Args:
        length: Length of the ID to generate.
        
    Returns:
        Random alphanumeric string.
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

def reformat_single_case(
    case: List[Dict[str, Any]],
    blue_threshold: float
) -> Optional[List[Dict[str, Any]]]:
    """Reformat a single trajectory case into the required format.
    
    Args:
        case: List of trajectory steps.
        blue_threshold: BLEU score threshold for rejecting similar outputs.
        
    Returns:
        List of reformatted cases or None if case is invalid.
    """
    try:
        images = list(set(case[0]['image_paths']))
    except (KeyError, AttributeError):
        logger.error("Image path error, skipping case")
        return None
        
    case_str = json.dumps(case, indent=4, ensure_ascii=False)
    if 'Internal Server Error' in case_str:
        logger.error("Internal Server Error, skipping case")
        return None
    if "Error in generating final llm output: CUDA out of memory." in case_str:
        logger.error("CUDA out of memory, skipping case")
        return None
        
    new_case_list = []
    for i in range(1, len(case)):
        try:
            best_step_idx = case[i]['best_step_idx']
            beam_size = case[i]['beam_size']
        except KeyError:
            logger.warning("Missing required fields, skipping step %d", i)
            continue
            
        # Construct reject candidate lists
        reject_candidates = []
        reject_observations = []
        for rj in range(beam_size):
            if rj != best_step_idx:
                try:
                    reject_candidates.append(case[i][f'llm_output_{rj}'])
                    reject_observations.append(case[i][f'observation_{rj}'])
                except KeyError:
                    logger.warning("Missing output for step %d, candidate %d", i, rj)
                    continue
                    
        if not reject_candidates:
            logger.warning("No reject candidates available, skipping step %d", i)
            continue
            
        # Process rejections
        error_reject_ids = get_error_case_reject_indices(reject_observations, reject_candidates)
        chosen_output = case[i][f'llm_output_{best_step_idx}']
        bleu_reject_ids = get_bleu_selected_indices(chosen_output, reject_candidates, blue_threshold)
        
        reject_ids = list(set(error_reject_ids[:2] + bleu_reject_ids))
        rejected_outputs = [reject_candidates[i] for i in reject_ids]
        rejected_observations = [reject_observations[i] for i in reject_ids]
        
        if 'Qwen Ranker error' in chosen_output:
            logger.warning("Qwen Ranker error, skipping step %d", i)
            continue
            
        new_case = {
            'images': images,
            'conversations': generate_conversations(case, i, images),
            'chosen': {
                'from': 'gpt',
                'value': chosen_output + '<end_action>',
                'observation': case[i][f'observation_{best_step_idx}']
            }
        }
        
        for rej_output, rej_obs in zip(rejected_outputs, rejected_observations):
            tmp_case = new_case.copy()
            tmp_case['rejected'] = {
                'from': 'gpt',
                'value': rej_output + '<end_action>',
                'observation': rej_obs
            }
            new_case_list.append(tmp_case)
            
    return new_case_list

def reformat_all_cases(cases: List[Dict[str, Any]], blue_threshold: float) -> List[Dict[str, Any]]:
    """Reformat all trajectory cases.
    
    Args:
        cases: List of trajectory cases.
        blue_threshold: BLEU score threshold for rejecting similar outputs.
        
    Returns:
        List of reformatted cases.
    """
    new_cases = []
    for case in cases:
        reformatted_case = reformat_single_case(case, blue_threshold)
        if reformatted_case is not None:
            new_cases.extend(reformatted_case)
    return new_cases

def load_system_prompt(prompt_path: str, image_path_prefix: str) -> str:
    """Load and process system prompt.
    
    Args:
        prompt_path: Path to the system prompt file.
        image_path_prefix: Prefix for image paths.
        
    Returns:
        Processed system prompt string.
    """
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompts = file.readlines()
    system_prompt = ''.join(prompts)
    return system_prompt.replace(".cache/", image_path_prefix)

def save_json(data: Any, save_path: str, file_name: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save.
        save_path: Directory to save the file in.
        file_name: Name of the file to save.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    with open(f'{save_path}{file_name}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def main() -> None:
    """Main function to process and save trajectory data."""
    args = parse_args()
    
    # Load and process cases
    cases = load_cases(args.traj_folder)
    logger.info("Loaded %d cases", len(cases))
    
    # Reformat cases
    new_cases = reformat_all_cases(cases, args.blue_threshold)
    logger.info("Reformatted %d cases", len(new_cases))
    
    # Add system prompt
    system_prompt = load_system_prompt(args.prompt_path, args.image_path_prefix)
    for case in new_cases:
        case["system"] = system_prompt
    
    # Save results
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    file_name = f'ruled_blue_less{args.blue_threshold}_{current_time}-num-{len(new_cases)}-w-obv-w_system_prompt'
    save_json(new_cases, args.save_path, file_name)
    logger.info("Saved results to %s%s.json", args.save_path, file_name)

if __name__ == "__main__":
    main()


