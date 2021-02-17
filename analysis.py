import matplotlib.pyplot as plt
import json
import os
import re
from data_utils.data_utils import get_multiwoz_config

slot_list = [
    "taxi-leaveAt",
    "taxi-destination",
    "taxi-departure",
    "taxi-arriveBy",
    "restaurant-book_people",
    "restaurant-book_day",
    "restaurant-book_time",
    "restaurant-food",
    "restaurant-pricerange",
    "restaurant-name",
    "restaurant-area",
    "hotel-book_people",
    "hotel-book_day",
    "hotel-book_stay",
    "hotel-name",
    "hotel-area",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-internet",
    "hotel-type",
    "attraction-type",
    "attraction-name",
    "attraction-area",
    "train-book_people",
    "train-leaveAt",
    "train-destination",
    "train-day",
    "train-arriveBy",
    "train-departure",
]


def tokenize(text):
    if "\u0120" in text:
        text = re.sub(" ", "", text)
        text = re.sub("\u0120", " ", text)
        text = text.strip()
    if text[:4] == "the ":
        text = text[4:]
    return " ".join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i : i + value_len] == value_list:
            found = True
            break
    return found


def calculate_joint_slot_acc(result_path, data_path="data/MULTIWOZ2.1/"):
    _, value_variations, _ = get_multiwoz_config()

    tokenized_value_variations = {}
    for val in value_variations:
        tokenized_value_variations[tokenize(val)] = [tokenize(v) for v in value_variations[val]]
    value_variations = tokenized_value_variations
    inverse_value_variations = {vv: k for k, v in value_variations.items() for vv in v}

    with open(result_path) as f:
        res = json.load(f)

    joint_correct, joint_total = 0, 0

    for r in res:
        all_correct = True
        if r["guid"].split("-")[-1] == "0":
            pred_dialogue_state = {slot: "none" for slot in slot_list}
            ground_truth_dialogue_state = {slot: "none" for slot in slot_list}
        for slot in slot_list:
            pred_val = tokenize(r[f"pred_value_{slot}"])
            gt_vals = []
            # for each ground truth value (in mw2.2 there are multiple),
            #   tokenize the given value, tokenize any variations of the value
            for gt_val in r[f"ground_truth_value_{slot}"]:
                tokenized_gt_val = tokenize(gt_val)
                gt_vals.append(tokenize(tokenized_gt_val))

                # add any variations
                if tokenized_gt_val in value_variations:
                    gt_vals.extend(value_variations[tokenized_gt_val])
                # add any variations
                if tokenized_gt_val in inverse_value_variations:
                    gt_vals.append(inverse_value_variations[tokenized_gt_val])
                    # ensure that we get all variations
                    gt_vals.extend(value_variations[inverse_value_variations[tokenized_gt_val]])

            pred_sources = r[f"pred_sources_{slot}"]
            gt_source = r[f"ground_truth_sources_{slot}"]

            match = pred_val in gt_vals

            if pred_sources == "none":
                pass
            else:
                pred_dialogue_state[slot] = pred_sources

            if gt_source == "none" or gt_source[0] == "none":
                pass
            else:
                ground_truth_dialogue_state[slot] = gt_source[0]

            cur_turn_inform = pred_dialogue_state[slot] == "inform" and ground_truth_dialogue_state[slot] == "inform"
            cur_turn_refer = pred_dialogue_state[slot] == "refer" and ground_truth_dialogue_state[slot] == "refer"

            if not match and (cur_turn_inform or cur_turn_refer):
                for gt_val in gt_vals:
                    if is_in_list(pred_val, gt_val) or is_in_list(gt_val, pred_val):
                        match = True
                        break
                    if pred_val in value_variations:
                        for pred_variation in value_variations[pred_val]:
                            if is_in_list(pred_variation, gt_val) or is_in_list(gt_val, pred_variation):
                                match = True
                                break
                    if pred_val in inverse_value_variations:
                        for pred_variation in inverse_value_variations[pred_val]:
                            if is_in_list(pred_variation, gt_val) or is_in_list(gt_val, pred_variation):
                                match = True
                                break
            # check if the predicted value is equivalent to the ground truth
            if not match:
                all_correct = False
                break

        if all_correct:
            joint_correct += 1
        joint_total += 1

    return joint_correct, joint_total


def calculate_joint_slot_accs_by_experiment(output_dirs):
    res = {}
    for output_dir in output_dirs:
        res[output_dir] = []
        ordered_dirs = sorted(os.listdir(output_dir), key=lambda x: int(x.split("-")[1]))
        for folder in ordered_dirs:
            subdir = os.path.join(output_dir, folder)
            if os.path.isdir(subdir):
                path = os.path.join(subdir, "predictions.json")
                res[output_dir].append([path, calculate_joint_slot_acc(path)])
                # print(f"{path} - {calculate_joint_slot_acc(path)}")
    return res


def plot_joint_slot_accs_by_experiment(output_dirs):
    res = calculate_joint_slot_accs_by_experiment(output_dirs)
    for path in res.keys():
        plt.plot([int(r[0].split("-")[2].split("/")[0]) for r in res[path]], [r[1][0] for r in res[path]], label=path)

    plt.legend()
    plt.show()


def analyze_errors(
    checkpoint_dir,
    prediction_source="val",
    data_path="data/MULTIWOZ2.1/",
    sources=["none", "dontcare", "usr_utt", "sys_utt", "inform", "refer", "DB", "true", "false"],
):
    # load predictions file
    result_path = os.path.join(checkpoint_dir, f"{prediction_source}_predictions.json")
    output_path = os.path.join(checkpoint_dir, f"{prediction_source}_error_analysis.json")
    with open(result_path) as f:
        res = json.load(f)

    # load value variations
    config_file = os.path.join(data_path, "config.json")
    with open(config_file, "r") as f:
        raw_config = json.load(f)

    # tokenize value variations and get inverse mapping
    value_variations = raw_config["label_maps"]
    tokenized_value_variations = {}
    for val in value_variations:
        tokenized_value_variations[tokenize(val)] = [tokenize(v) for v in value_variations[val]]
    value_variations = tokenized_value_variations
    inverse_value_variations = {vv: k for k, v in value_variations.items() for vv in v}

    analysis = {}

    overall_source_scores = {source: {"correct": 0, "total": 0} for source in sources}

    for slot in slot_list:
        slot_info = {"incorrect_predictions": []}
        source_scores_by_slot = {source: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for source in sources}
        for r in res:
            if r["guid"].split("-")[-1] == "0":
                pred_dialogue_state = "none"
                ground_truth_dialogue_state = "none"

            pred_val = tokenize(r[f"pred_value_{slot}"])
            gt_vals = set()
            # for each ground truth value (in mw2.2 there are multiple),
            #   tokenize the given value, tokenize any variations of the value
            for gt_val in r[f"ground_truth_value_{slot}"]:
                tokenized_gt_val = tokenize(gt_val)
                gt_vals.add(tokenized_gt_val)

                # add any variations
                if tokenized_gt_val in value_variations:
                    for val in value_variations[tokenized_gt_val]:
                        gt_vals.add(val)
                # add any variations
                if tokenized_gt_val in inverse_value_variations:
                    gt_vals.add(inverse_value_variations[tokenized_gt_val])
                    for val in value_variations[inverse_value_variations[tokenized_gt_val]]:
                        gt_vals.add(val)

            pred_sources = r[f"pred_sources_{slot}"]
            gt_source = r[f"ground_truth_sources_{slot}"]

            match = pred_val in gt_vals

            if pred_sources == "none":
                pass
            else:
                pred_dialogue_state = pred_sources

            if gt_source == "none" or gt_source[0] == "none":
                pass
            else:
                ground_truth_dialogue_state = gt_source[0]

            cur_turn_inform = pred_dialogue_state == "inform" and ground_truth_dialogue_state == "inform"
            cur_turn_refer = pred_dialogue_state == "refer" and ground_truth_dialogue_state == "refer"

            if not match and (cur_turn_inform or cur_turn_refer):
                for gt_val in gt_vals:
                    if is_in_list(pred_val, gt_val) or is_in_list(gt_val, pred_val):
                        match = True
                        break
                    if pred_val in value_variations:
                        for pred_variation in value_variations[pred_val]:
                            if is_in_list(pred_variation, gt_val) or is_in_list(gt_val, pred_variation):
                                match = True
                                break
                    if pred_val in inverse_value_variations:
                        for pred_variation in inverse_value_variations[pred_val]:
                            if is_in_list(pred_variation, gt_val) or is_in_list(gt_val, pred_variation):
                                match = True
                                break

            if not match:
                slot_info["incorrect_predictions"].append(
                    f"{r['guid']} - PREDICTED {pred_val} (source {pred_sources}) | GROUND TRUTH {gt_vals} (source {gt_source}"
                )

            # calculate tp, fp, fn, tn scores per source
            for source in sources:
                if source in gt_source:
                    if source in pred_sources:
                        source_scores_by_slot[source]["tp"] += 1
                    else:
                        source_scores_by_slot[source]["fn"] += 1
                else:
                    if source in pred_sources:
                        source_scores_by_slot[source]["fp"] += 1
                    else:
                        source_scores_by_slot[source]["tn"] += 1

            # calculate overall source accuracy
            for source in gt_source:
                overall_source_scores[source]["total"] += 1
                if source in pred_sources:
                    overall_source_scores[source]["correct"] += 1

        slot_info["source_performance_by_slot"] = source_scores_by_slot
        analysis[slot] = slot_info

    analysis["overall_source_scores"] = overall_source_scores
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)