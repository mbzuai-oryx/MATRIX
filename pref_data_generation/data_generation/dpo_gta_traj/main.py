"""Main script for DPO GTA trajectory data generation.

This script handles the generation of trajectory data using DPO agents with support for
both text-only and multimodal tasks. It includes task management, parallel processing,
and result saving functionality.
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Any

import ray
import torch
from filelock import FileLock
from tqdm import tqdm

from tongagent.agents.dpo_agent_data_sampling import create_agent
from tongagent.utils import load_config

# Constants
TASK_FILE = "data_generation/dpo_gta_traj/record/task_info.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="DPO GTA trajectory data generation script"
    )
    
    parser.add_argument(
        "--engine",
        "-e",
        choices=["tonggpt", "qwen"],
        default="tonggpt",
        help="LLM engine to use"
    )
    
    parser.add_argument(
        "--lora-path",
        "-lp",
        default=None,
        help="Path to LoRA weights"
    )
    
    parser.add_argument(
        "--disable-vision",
        action="store_true",
        help="Disable vision capabilities"
    )
    
    parser.add_argument(
        "--dpo-agent",
        action="store_true",
        help="Use DPOAgent to evaluate results"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Size of parallel sampling"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index for processing"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=800,
        help="Ending index for processing"
    )
    
    parser.add_argument(
        "--verifier",
        type=str,
        default="best_selector",
        help="Verifier to use"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="data_generation/dpo_gta_traj/source/multimodal_tasks_num1107.json",
        help="Source data file path"
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        default="data_generation/dpo_gta_traj/save",
        help="Path to save results"
    )
    
    parser.add_argument(
        "--lock",
        type=str,
        default="data_generation/dpo_gta_traj/record/tasks.lock",
        help="Lock file path"
    )
    
    return parser.parse_args()


def generate_random_filename() -> str:
    """Generate a random filename for saving results.
    
    Returns:
        str: Randomly generated filename
    """
    random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
    random_num = ''.join(random.choices('0123456789', k=5))
    return f"{random_str}_{random_num}.json"


def get_task(lock_file: str) -> Optional[int]:
    """Get the next task from the task queue.
    
    Args:
        lock_file: Path to the lock file
        
    Returns:
        Optional[int]: Task ID if available, None otherwise
    """
    with FileLock(lock_file):
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data["remaining"]:
            return None
        
        task_id = data["remaining"].pop(0)
        data["generating"].append(task_id)

        with open(TASK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        logging.info("Task %d assigned successfully", task_id)
        logging.info("Remaining tasks: %d", len(data["remaining"]))
        logging.info("Tasks in progress: %s", data["generating"])
        logging.info("Completed tasks: %s", data["completed"])

    return task_id


def complete_task(task_id: int, lock_file: str) -> int:
    """Mark a task as completed.
    
    Args:
        task_id: ID of the completed task
        lock_file: Path to the lock file
        
    Returns:
        int: Task ID
    """
    with FileLock(lock_file):
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_id in data["generating"]:
            data["generating"].remove(task_id)
            data["completed"].append(task_id)
        
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    return task_id


def failed_task(task_id: int, lock_file: str) -> int:
    """Mark a task as failed and requeue it.
    
    Args:
        task_id: ID of the failed task
        lock_file: Path to the lock file
        
    Returns:
        int: Task ID
    """
    with FileLock(lock_file):
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_id in data["generating"]:
            data["generating"].remove(task_id)
            data["failed"].append(task_id)
            data["remaining"].append(task_id)
        
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    return task_id


def process_task(
    task_id: int,
    item: Dict[str, Any],
    agent: Any,
    root_path: str,
    lock_file: str
) -> None:
    """Process a single task.
    
    Args:
        task_id: ID of the task to process
        item: Task data
        agent: DPO agent instance
        root_path: Root path for saving results
        lock_file: Path to the lock file
    """
    try:
        question = item["query"]
        if "files" not in item:
            logging.info("Task %d has no images", task_id)
            question = f"{question}\n"
            agent.set_image_paths([])
            agent.set_captions([])
        else:
            image_paths = [
                os.path.join("data/tongagent-ablation-1k", i["path"])
                for i in item["files"]
            ]
            attachment = "; ".join(image_paths)
            question = f"{question}\n Attachment: {attachment}"
            captions = [i["caption"] for i in item["files"]]
            agent.set_image_paths(image_paths)
            agent.set_captions(captions)

        result = agent.run(question)
        agent.save_trajectory(path=root_path, final_answer=result)
        complete_task(task_id, lock_file)
        logging.info("Task %d completed successfully", task_id)
        
    except Exception as e:
        logging.error("Error processing task %d: %s", task_id, str(e))
        failed_task(task_id, lock_file)


def worker(dataset: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Worker function for processing tasks.
    
    Args:
        dataset: List of tasks to process
        args: Command line arguments
    """
    agent = create_agent(
        llm_engine=args.engine,
        task="gta",
        error_tolerance=5,
        lora_path=args.lora_path,
        disable_vision=args.disable_vision,
        sampling_size=args.sample,
    )
    
    while True:
        task_id = get_task(args.lock)
        if task_id is None:
            logging.info("All tasks completed")
            break
            
        logging.info("Processing task %d", task_id)
        process_task(task_id, dataset[task_id], agent, args.save_path, args.lock)


@ray.remote(num_gpus=1)
def remote_worker(dataset: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Remote worker function for parallel processing.
    
    Args:
        dataset: List of tasks to process
        args: Command line arguments
    """
    worker(dataset, args)


def main() -> None:
    """Main function."""
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    
    with open(args.source, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    config = load_config()
    
    lora_path = args.lora_path if args.lora_path else "default"
    root = os.path.join(args.save_path, lora_path.split("/")[-1])
    
    if args.disable_vision:
        root += "_without_vision"
    
    os.makedirs(root, exist_ok=True)
    logging.info("Saving results to %s", root)
    
    ray.init()
    
    n_total_gpu = torch.cuda.device_count()
    n_total_data = len(dataset)
    
    if n_total_gpu == 1:
        worker(dataset, args)
    else:
        futures = []
        for _ in range(n_total_gpu):
            futures.append(remote_worker.remote(dataset, args))
        
        ray.get(futures)
    
    ray.shutdown()


if __name__ == "__main__":
    main()