import logging
import os
import json
from tqdm import tqdm

import utils.utils
from create_data import get_data

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            # if element does not contain tensors, do not move it to device
            if isinstance(list(element.values())[0], torch.Tensor):
                batch_on_device.append({k: v.to(device) for k, v in element.items()})
            else:
                batch_on_device.append(element)
        elif isinstance(element[0], str):
            batch_on_device.append(element[0])
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)


def main(**kwargs):
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    model_path = os.path.join(kwargs["output_dir"], kwargs["eval_name"])

    # define model variables
    set_seed(kwargs["seed"])
    kwargs["model_type"] = kwargs["model_type"].lower()
    config_class, model_class, tokenizer_class = utils.utils.MODEL_CLASSES[kwargs["model_type"]]
    config = config_class.from_pretrained(model_path)

    tokenizer = tokenizer_class.from_pretrained(kwargs["model_name_or_path"], do_lower_case=kwargs["do_lower_case"])

    model = model_class.from_pretrained(model_path)
    model.to(kwargs["device"])
    model.eval()

    if kwargs["validation"]:
        kwargs["dataset_type"] = "val"
    if kwargs["test"]:
        kwargs["dataset_type"] = "test"
    if kwargs["debugging"]:
        kwargs["dataset_type"] = "debugging"

    test_dataset, test_features = get_data(**kwargs)
    # batch_size always set to 1 to handle sequential nature of dialogue
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    logger.info(f"******** Evaluating {kwargs['eval_name']} ************")
    logger.info(f"  Num samples: {len(test_dataset)}")

    predictions = []
    pred_dialog_state = {slot: "none" for slot in model.slot_list}

    for batch in tqdm(test_dataloader):
        batch = batch_to_device(batch, kwargs["device"])
        guid = batch[0]
        input_ids = batch[1]
        attention_mask = batch[2]
        segment_ids = batch[3]
        start_labels = batch[4]
        end_labels = batch[5]
        seen_values = batch[6]
        values = batch[7]
        value_sources = batch[8]
        dialog_states = batch[9]
        inform_values = batch[10]
        inform_slot_labels = batch[11]
        refer_labels = batch[12]
        DB_labels = batch[13]

        # if this is the first turn in a dialogue, reset dialogue state
        reset_dialog_state = guid.split("-")[2] == "0"
        if reset_dialog_state:
            for slot in model.slot_list:
                pred_dialog_state[slot] = "none"

        with torch.no_grad():
            accuracies, pred_dialog_state, sample_info = model.evaluate(
                tokenizer,
                guid,
                input_ids,
                attention_mask,
                segment_ids,
                start_labels,
                end_labels,
                seen_values,
                values,
                value_sources,
                dialog_states,
                pred_dialog_state,
                inform_values,
                inform_slot_labels,
                refer_labels,
                DB_labels,
                kwargs["softgate"],
            )
        predictions.append(sample_info)

    output_predictions_file = os.path.join(model_path, "predictions.json")
    with open(output_predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)


if __name__ == "__main__":
    args = utils.utils.parse_args()
    main(**args)