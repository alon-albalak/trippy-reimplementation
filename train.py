import logging
import os
import json
from tqdm import tqdm

import utils.utils
from analysis import calculate_joint_slot_acc
from create_data import get_data

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

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

    # define model variables
    if kwargs["seed"] != -1:
        set_seed(kwargs["seed"])
    kwargs["model_type"] = kwargs["model_type"].lower()
    config_class, model_class, tokenizer_class = utils.utils.MODEL_CLASSES[kwargs["model_type"]]
    config = config_class.from_pretrained(kwargs["model_name_or_path"])

    config.slot_list = kwargs["slot_list"]
    config.sources = kwargs["sources"]
    config.bert_dropout_rate = kwargs["bert_dropout_rate"]
    config.source_loss_ratio = kwargs["source_loss_ratio"]
    config.downweight_none_slot = kwargs["downweight_none_slot"]

    tokenizer = tokenizer_class.from_pretrained(kwargs["model_name_or_path"], do_lower_case=kwargs["do_lower_case"])

    model = model_class.from_pretrained(kwargs["model_name_or_path"], config=config)
    model.to(kwargs["device"])
    model.train()

    # load dataset into dataloader
    if kwargs["debugging"]:
        kwargs["dataset_type"] = "debugging"
    else:
        kwargs["dataset_type"] = "train"
    train_dataset, train_features = get_data(**kwargs)
    train_dataloader = DataLoader(
        train_dataset, batch_size=kwargs["gpu_batch_size"], shuffle=True, num_workers=kwargs["num_workers"], pin_memory=kwargs["pin_memory"]
    )
    if kwargs["eval_during_training"]:
        if kwargs["debugging"]:
            kwargs["dataset_type"] = "train_debugging"
        else:
            kwargs["dataset_type"] = "val"
        val_dataset, val_features = get_data(**kwargs)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=kwargs["pin_memory"])

    gradient_accumulation_steps = kwargs["effective_batch_size"] / kwargs["gpu_batch_size"]
    total_optimization_steps = kwargs["num_epochs"] * (len(train_dataloader) // gradient_accumulation_steps)
    num_warmup_steps = total_optimization_steps * kwargs["warmup_proportion"]

    optimizer = AdamW(model.parameters(), lr=kwargs["learning_rate"], eps=kwargs["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_optimization_steps)

    if kwargs["fp16"]:
        scaler = GradScaler()

    logger.info("********* TRAINING ************")
    logger.info(f"   Num samples: {len(train_dataset)}")
    logger.info(f"   Num epochs: {kwargs['num_epochs']}")
    logger.info(f"   Total optimization steps: {total_optimization_steps}")
    logger.info(f"   Warmup steps {num_warmup_steps}")

    best_joint_acc = 0

    for epoch in range(kwargs["num_epochs"]):
        # train loop
        logger.info(f"  Epoch: {epoch+1}")

        total_loss = 0
        total_source_loss = 0
        total_token_loss = 0
        total_refer_loss = 0
        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        # iterate over batches
        for step, batch in pbar:
            batch = batch_to_device(batch, kwargs["device"])
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

            if kwargs["fp16"]:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        segment_ids=segment_ids,
                        dialog_states=dialog_states,
                        inform_slot_labels=inform_slot_labels,
                    )
                    per_slot_source_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits = outputs

                    loss, source_loss, token_loss, refer_loss, source_acc, start_acc, end_acc, refer_acc = model.calculate_loss(
                        attention_mask=attention_mask,
                        start_labels=start_labels,
                        end_labels=end_labels,
                        value_sources=value_sources,
                        refer_labels=refer_labels,
                        DB_labels=DB_labels,
                        per_slot_source_logits=per_slot_source_logits,
                        per_slot_start_logits=per_slot_start_logits,
                        per_slot_end_logits=per_slot_end_logits,
                        per_slot_refer_logits=per_slot_refer_logits,
                    )
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if ((step + 1) % gradient_accumulation_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    dialog_states=dialog_states,
                    inform_slot_labels=inform_slot_labels,
                )

                per_slot_source_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits = outputs

                loss, source_loss, token_loss, refer_loss, source_acc, start_acc, end_acc, refer_acc = model.calculate_loss(
                    attention_mask=attention_mask,
                    start_labels=start_labels,
                    end_labels=end_labels,
                    value_sources=value_sources,
                    refer_labels=refer_labels,
                    DB_labels=DB_labels,
                    per_slot_source_logits=per_slot_source_logits,
                    per_slot_start_logits=per_slot_start_logits,
                    per_slot_end_logits=per_slot_end_logits,
                    per_slot_refer_logits=per_slot_refer_logits,
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["max_grad_norm"])
                if ((step + 1) % gradient_accumulation_steps) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            total_loss += loss.item()
            with torch.no_grad():
                total_source_loss += source_loss.item() / gradient_accumulation_steps
                total_token_loss += token_loss.item() / gradient_accumulation_steps
                total_refer_loss += refer_loss.item() / gradient_accumulation_steps

            desc = f"TRAIN Loss: {total_loss/(step+1):0.4f} === src loss: {total_source_loss/(step+1):0.4f} === token loss: {total_token_loss/(step+1):0.4f} === refer loss: {total_refer_loss/(step+1):0.4f}"
            if kwargs["calculate_accs"]:
                desc += f" === source acc {source_acc*100:0.2f} === start acc {start_acc*100:0.2f} === end acc {end_acc*100:0.2f} === refer acc {refer_acc*100:0.2f}"
            pbar.set_description(desc)

        # # save model checkpoints
        if kwargs["save_model_checkpoints"]:
            output_dir = os.path.join(kwargs["output_dir"], f"checkpoint-{epoch+1}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            logger.info(f"    Saving model checkpoint to {output_dir}")

        if kwargs["eval_during_training"]:
            predictions = []
            pred_dialog_state = {slot: "none" for slot in model.slot_list}
            source_acc, token_acc, refer_acc = 1, 1, 1

            model.eval()
            pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            for step, batch in pbar:
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
                    if kwargs["fp16"]:
                        with autocast():
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
                                kwargs["compute_full_value_distribution"],
                            )
                    else:
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
                            kwargs["compute_full_value_distribution"],
                        )
                predictions.append(sample_info)
                # pbar.set_description(
                #     f"VAL Loss: {total_loss/(step+1):0.4f} === src loss: {total_source_loss/(step+1):0.4f} === token loss: {total_token_loss/(step+1):0.4f} === refer loss: {total_refer_loss/(step+1):0.4f}"
                # )
            output_dir = os.path.join(kwargs["output_dir"], f"checkpoint-{epoch+1}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_prediction_file = os.path.join(output_dir, "predictions.json")
            with open(output_prediction_file, "w") as f:
                json.dump(predictions, f, indent=2)
            joint_correct, joint_total = calculate_joint_slot_acc(output_prediction_file)
            joint_slot_acc = joint_correct / joint_total
            logger.info(f"    Joint Slot Accuracy - checkpoint-{epoch+1} - {joint_slot_acc:0.3f}")
            os.rename(output_dir, f"{output_dir}-{joint_slot_acc:0.3f}")


if __name__ == "__main__":
    args = utils.utils.parse_args()

    if args["num_workers"] > 0:
        torch.multiprocessing.set_start_method("spawn")

    main(**args)