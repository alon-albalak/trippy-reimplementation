import os
import json
import re
import logging
from tqdm import tqdm
import utils.utils
from data_utils.data_utils import convert_examples_to_features, TensorListDataset
from data_utils.multiwoz21 import load_multiwoz21_dataset
import torch

logger = logging.getLogger(__name__)


def create_data(cached_file, **kwargs):

    logger.info(f"******* Creating {kwargs['dataset_identifier']} Data ***********")
    model_config, model, model_tokenizer = utils.utils.MODEL_CLASSES[kwargs["model_type"]]
    tokenizer = model_tokenizer.from_pretrained(kwargs["model_name_or_path"], do_lower_case=kwargs["do_lower_case"])
    data = load_multiwoz21_dataset(
        kwargs["dataset_type"],
        kwargs["label_value_repetitions"],
        kwargs["label_only_last_occurence"],
        kwargs['single_source'],
        kwargs['delexicalize_sys_utt'],
        kwargs["data_path"],
        kwargs["DB_file"],
        kwargs["sources"],
        kwargs["log_unpointable_values"],
    )
    features = convert_examples_to_features(
        data, kwargs["slot_list"], kwargs["model_type"], tokenizer, kwargs["max_sequence_len"], kwargs["exact_reimplementation"]
    )
    if kwargs["cache_features"]:
        logger.info(f"Saving features into cached file {cached_file}")
        torch.save(features, cached_file)

    return features


def get_data(**kwargs):
    """Create or load data, if it exists"""
    cached_file = os.path.join(os.path.dirname(kwargs["output_dir"]), f"cached_{kwargs['dataset_type']}_features")
    if kwargs["dataset_identifier"]:
        cached_file += f"_{kwargs['dataset_identifier']}"
    if os.path.exists(cached_file) and not kwargs["overwrite_cache"]:
        logger.info(f"Loading features from {cached_file}")
        features = torch.load(cached_file)
    else:
        features = create_data(cached_file, **kwargs)

    # convert to Tensors and add to Dataset
    all_guids = [f.guid for f in features]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)  # Tensor of dim: Num_examples x sequence_len

    # reorganize all slot-dependent features into list first
    f_start_labels = [f.start_labels for f in features]  # list of dicts, each dict is: {slot: sequence_len list of 0/1's}
    f_end_labels = [f.end_labels for f in features]
    f_seen_values = [f.seen_values for f in features]
    f_values = [f.values for f in features]
    f_value_sources = [f.value_sources for f in features]  # list of dicts, each dict is {slot: num_sources len list of 0/1's}
    f_dialog_states = [f.dialog_states for f in features]
    f_inform_values = [f.inform_values for f in features]
    f_inform_slot_labels = [f.inform_slot_labels for f in features]
    f_refer_labels = [f.refer_labels for f in features]

    all_start_labels = {}
    all_end_labels = {}
    all_seen_values = {}
    all_values = {}
    all_value_sources = {}
    all_dialog_states = {}
    all_inform_values = {}
    all_inform_slot_labels = {}
    all_refer_labels = {}

    if "DB" in kwargs["sources"]:
        f_DB_labels = [f.DB_labels for f in features]
        all_DB_labels = {}

    # reorganize all slot-dependent features by slot
    for slot in kwargs["slot_list"]:

        if kwargs['single_token_label']:
            # convert token labels to index notation
            for f in f_start_labels:
                if sum(f[slot]) == 0:
                    f[slot] = 0
                else:
                    f[slot] = f[slot].index(1)
            for f in f_end_labels:
                if sum(f[slot]) == 0:
                    f[slot] = 0
                else:
                    f[slot] = f[slot].index(1)

        if kwargs["single_source"]:
            # for value sources, and dialogue state convert to index notation
            for vs, ds in zip(f_value_sources, f_dialog_states):
                vs[slot] = kwargs["sources"].index(vs[slot])
                ds[slot] = kwargs["sources"].index(ds[slot])

        all_start_labels[slot] = torch.tensor(
            [f[slot] for f in f_start_labels], dtype=torch.long if kwargs["exact_reimplementation"] else torch.float
        )
        all_end_labels[slot] = torch.tensor([f[slot] for f in f_end_labels], dtype=torch.long if kwargs["exact_reimplementation"] else torch.float)
        all_seen_values[slot] = [f[slot] for f in f_seen_values]
        all_values[slot] = [f[slot] for f in f_values]
        all_value_sources[slot] = torch.tensor(
            [f[slot] for f in f_value_sources], dtype=torch.long if kwargs["exact_reimplementation"] else torch.float
        )
        all_dialog_states[slot] = torch.tensor([f[slot] for f in f_dialog_states], dtype=torch.float)
        all_inform_values[slot] = [f[slot] for f in f_inform_values]
        all_inform_slot_labels[slot] = torch.tensor([f[slot] for f in f_inform_slot_labels], dtype=torch.float)
        all_refer_labels[slot] = torch.tensor([f[slot] for f in f_refer_labels], dtype=torch.long)
        if "DB" in kwargs["sources"]:
            all_DB_labels[slot] = torch.tensor([f[slot] for f in f_DB_labels], dtype=torch.long)
        else:
            all_DB_labels = []

    dataset = TensorListDataset(
        all_guids,
        all_input_ids,
        all_attention_mask,
        all_segment_ids,
        all_start_labels,
        all_end_labels,
        all_seen_values,
        all_values,
        all_value_sources,
        all_dialog_states,
        all_inform_values,
        all_inform_slot_labels,
        all_refer_labels,
        all_DB_labels,
    )

    return dataset, features


if __name__ == "__main__":
    args = utils.utils.parse_args()

    create_data(**args)