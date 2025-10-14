import json
import argparse
import os
import re

def gt_answer_augment(gts):
    gts_processed = []
    for gt in gts:
        assert type(gt) is list
        gt_processed = []
        for gt_item in gt:
            gt_processed.append(gt_item.lower())
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
        gts_processed.append(gt_processed)
    return gts_processed



gt = [
    ["1000"]
]

print(gt_answer_augment(gt))


gt = [
    ["0"],
    ["1"]
]

print(gt_answer_augment(gt))

gt = [
    ["10%"],
    ["24G"]
]

print(gt_answer_augment(gt))