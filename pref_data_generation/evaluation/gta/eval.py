import json
import argparse
import os
import re
parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True)

args = parser.parse_args()
subfolders = os.listdir(args.folder)

total = 0
correct = 0
all_samples = len(subfolders)


def gt_answer_augment(gts):
    if type(gts) is str:
        return gts
    gts_processed = []
    for gt in gts:
        if type(gt) is str:
            return gts
        gt_processed = []
        for gt_item in gt:
            gt_processed.append(gt_item.lower())
            gt_processed.append(gt_item.upper())
            # case 1
            try:
                gt_item_processed = int(gt_item.lower())
                if gt_item_processed <= 10:
                    text_mapping = {
                        0: "zero",
                        1: "one",
                        2: "two",
                        3: "three",
                        4: "four",
                        5: "five",
                        6: "six",
                        7: "seven",
                        8: "eight",
                        9: "nine",
                        10: "ten",
                    }
                    gt_processed.append(text_mapping[gt_item_processed])
                    if gt_item_processed == 0:
                        gt_processed.append("no")
                else:
                    # if number has more than 3 digits, then process to a string with comma, eg 1000 -> 1,000
                    if gt_item_processed >= 1000:
                        gt_processed.append(f"{gt_item_processed // 1000},{gt_item_processed % 1000:03d}")
                    else:
                        gt_processed.append(str(gt_item_processed))
                continue # skip to next item
            except:
                print("No pure number value")
                pass
            
            
            # case 2: check if gt_item is a number followed by a single character or symbol
            if re.match(r'^\d+([a-zA-Z%]+)$', gt_item):
                print(f"Found number with string value: {gt_item}")
                # Extract number and symbol using regex groups
                number = re.match(r'^\d+', gt_item).group()
                symbol = re.search(r'[a-zA-Z%]+$', gt_item).group()
                gt_item_processed = int(number)
                gt_processed.append(f"{gt_item_processed} {symbol}")
                continue # skip to next item
            else:
                print("No number with string value")
                pass

            # if 'PM' in gt
            if 'PM' in gt_item:
                gt_processed.append(gt_item.replace('PM', 'p.m.'))
                gt_processed.append(gt_item.replace('PM', ' PM'))
                continue
        gts_processed.append(list(set(gt_processed)))
    return gts_processed

# for subfolder in subfolders:
#     data_path = os.path.join(args.folder, subfolder, "agent_memory.json")
#     eval_result_path = os.path.join(args.folder, subfolder, "eval_result.json")
#     with open(data_path, "r") as f:
#         try:
#             dataset = json.load(f)
#         except Exception as e:
#             print(f"Error: {e} for {data_path}")
#             try:
#                 data_path = os.path.join(args.folder, subfolder, "final_answer.json")
#                 with open(data_path, "r") as f:
#                     dataset = json.load(f)
#                     print(f"Loaded {data_path} successfully!")
#             except Exception as e:
#                 print(f"Error: {e} for {data_path}")
#                 with open(eval_result_path, "w") as f:
#                     json.dump({"gt": None, "answer": None, "is_correct": False, "message": str(e)}, f, indent=4, ensure_ascii=False)
#                 continue
# gt, answer = dataset["ground_truth"], dataset["final_answer"]          
for subfolder in subfolders:
    data_path = os.path.join(args.folder, subfolder, "beam_search_data.json")
    if not os.path.exists(data_path):
        continue
    with open(data_path, "r") as f:
        dataset = json.load(f)
        print(f"Loaded {data_path} successfully!")
        images =  dataset[0]["image_paths"]
        print("Images", images)
        answer = dataset[0]["final_answer"].replace("Final Answer: ", "")
    eval_result_path = os.path.join(args.folder, subfolder, "eval_result.json")
    if '.DS_Store' in data_path:
        continue
    with open(data_path, "r") as f:
        try:
            dataset = json.load(f)
            images =  dataset[0]["image_paths"]
            print("Images", images)
            answer = dataset[0]["final_answer"].replace("Final Answer: ", "")
            # print(dataset[0])

             
            try:
                data_path = os.path.join(args.folder, subfolder, "final_answer.json")
                with open(data_path, "r") as f:
                    dataset = json.load(f)
                    print(f"Loaded {data_path} successfully!")
                    gt = dataset["ground_truth"]
            except Exception as e:
                print(f"Error: {e} for {data_path}")
                pass
        except Exception as e:
            print(f"Error: {e} for {data_path}")
            try:
                data_path = os.path.join(args.folder, subfolder, "final_answer.json")
                with open(data_path, "r") as f:
                    dataset = json.load(f)
                    print(f"Loaded {data_path} successfully!")
            except Exception as e:
                print(f"Error: {e} for {data_path}")
                with open(eval_result_path, "w") as f:
                    json.dump({"gt": None, "answer": None, "is_correct": False, "message": str(e)}, f, indent=4, ensure_ascii=False)
                continue    
     
    print("GT", gt)
    print("ANSWER", answer) 

    if len(images) > 1:
        skip = True
    if gt is None:
        continue
    skip = False
    is_correct = True
    gt = gt_answer_augment(gt)
    # if 'CUDA out of memory' in answer:
    #     skip = True

    
    for each in gt:
        if type(each) is str:
            skip = True
            break

        
        if type(each) is list:
            is_this_gt_correct = []
            for item in each:
                is_this_gt_correct.append(item.lower() in str(answer).lower())
            
            
            is_correct = is_correct and any(is_this_gt_correct)
        else:
            raise ValueError("unexpected")       
    if skip:
        continue
    
    if is_correct:
        print("Correct:", gt, answer)
        correct += 1
    else:
        print("Incorrect", gt, answer)
    total += 1
    with open(eval_result_path, "w") as f:
        json.dump({"gt": gt, "answer": answer, "is_correct": is_correct}, f, indent=4, ensure_ascii=False)
        
    # print(gt, answer)

print("Folder", args)
print("Total samples valid:", total, "Correct sample", correct, "all samples", all_samples)
print("Accuracy", round(correct / total, 4) * 100, "%")
print("Accuracy (all samples)", round(correct / all_samples, 4) * 100, "%")