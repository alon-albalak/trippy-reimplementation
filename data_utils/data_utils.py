import logging
import json
import re
from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def decode_normalize_tokens(input_tokens, start_idx, end_idx):
    pred_value = " ".join(input_tokens[start_idx : end_idx + 1])
    pred_value = re.sub("(^| )##", "", pred_value)
    return pred_value


def combine_value_variations_in_value_predictions(value_prediction_distribution, value_variations, inverse_value_variations):
    new_value_predictions = {}
    for val in value_prediction_distribution:
        found = False
        if val in new_value_predictions:
            new_value_predictions[val] += value_prediction_distribution[val]
            found = True
        elif val in value_variations or val in inverse_value_variations:
            if val in value_variations:
                for variation in value_variations[val]:
                    if variation in new_value_predictions:
                        new_value_predictions[variation] += value_prediction_distribution[val]
                        found = True
                        break
            if found:
                break
            if val in inverse_value_variations:
                if inverse_value_variations[val] in new_value_predictions:
                    new_value_predictions[inverse_value_variations[val]] += value_prediction_distribution[val]
                    found = True

        if not found:
            new_value_predictions[val] = value_prediction_distribution[val]
    return new_value_predictions


def get_multiwoz_config(config_file="data/MULTIWOZ2.1/config.json"):
    with open(config_file, "r") as f:
        raw_config = json.load(f)
    slot_list = raw_config["slots"]
    value_variations = raw_config["label_maps"]
    inverse_value_variations = {vv: k for k, v in value_variations.items() for vv in v}
    return slot_list, value_variations, inverse_value_variations


class Example(object):
    """
    A single training/test example for the Multiwoz2.1 dataset.
    """

    def __init__(
        self,
        guid,
        value_sources,
        usr_utterance_tokens,
        sys_utterance_tokens,
        history,
        usr_utterance_token_label_dict=None,
        sys_utterance_token_label_dict=None,
        hst_utterance_token_label_dict=None,
        seen_values=None,
        values=None,
        inform_value=None,
        inform_slot_label=None,
        refer_label=None,
        dialog_states=None,
        DB_label=None,
    ):
        self.guid = guid
        self.value_sources = value_sources
        self.usr_utterance_tokens = usr_utterance_tokens
        self.sys_utterance_tokens = sys_utterance_tokens
        self.history = history
        self.usr_utterance_token_label_dict = usr_utterance_token_label_dict
        self.sys_utterance_token_label_dict = sys_utterance_token_label_dict
        self.hst_utterance_token_label_dict = hst_utterance_token_label_dict
        self.seen_values = seen_values
        self.values = values
        self.inform_value = inform_value
        self.inform_slot_label = inform_slot_label
        self.refer_label = refer_label
        self.dialog_states = dialog_states
        self.DB_label = DB_label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "guid: %s" % (self.guid)
        s += "value_sources: %s" % (self.value_sources)
        s += ", usr_utterance_tokens: %s" % (self.usr_utterance_tokens)
        s += ", sys_utterance_tokens: %s" % (self.sys_utterance_tokens)
        s += ", history: %s" % (self.history)
        if self.usr_utterance_token_label_dict:
            s += ", usr_utterance_token_label_dict: %s" % (self.usr_utterance_token_label_dict)
        if self.sys_utterance_token_label_dict:
            s += ", sys_utterance_token_label_dict: %s" % (self.sys_utterance_token_label_dict)
        if self.hst_utterance_token_label_dict:
            s += ", hst_utterance_token_label_dict: %s" % (self.hst_utterance_token_label_dict)
        if self.values:
            s += ", values: %s" % (self.values)
        if self.inform_value:
            s += ", inform_value: %s" % (self.inform_value)
        if self.inform_slot_label:
            s += ", inform_slot_label: %s" % (self.inform_slot_label)
        if self.refer_label:
            s += ", refer_label: %s" % (self.refer_label)
        if self.dialog_state:
            s += ", dialog_state: %d" % (self.dialog_state)
        # if self.class_label:
        #     s += ", class_label: %d" % (self.class_label)
        if self.DB_label:
            s += ", DB_label: %d" % (self.DB_label)
        return s


class Features(object):
    """A single set of features of data"""

    def __init__(
        self,
        guid,
        input_ids,  # tokens for BERT model
        attention_mask,  # token mask for BERT model to attend to, 1 for tokens to attend to, 0 for other tokens, also used for padding sequences
        segment_ids,  # segment ids for BERT model
        start_labels,  # binary label for each token, can be 0, 1, or multiple start positions
        end_labels,  # binary label for each token, can be 0, 1 or multiple end positions
        seen_values,  # dict of {slot: value}
        values,  # dict of {slot: value}
        value_sources,  # dict of {slot: [list of sources]} where each source is binary and there may be more than 1 source
        dialog_states,  # dict of {slot: [list of sources]}
        inform_values,  # dict of {slot: informed value}
        inform_slot_labels,  # dict of {slot: inform_value} for each slot, where label is either 0 or 1
        refer_labels,  # dict of {slot: referred_slot_label} for each slot, where referred slot is the index of the slot being referred to
        DB_labels,  # dict of {slot: DB label} for each slot, where DB_label is the index of the value in the DB
    ):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.seen_values = seen_values
        self.values = values
        self.value_sources = value_sources
        self.dialog_states = dialog_states
        self.inform_values = inform_values
        self.inform_slot_labels = inform_slot_labels
        self.refer_labels = refer_labels
        self.DB_labels = DB_labels


def convert_examples_to_features(examples, slot_list, model_type, tokenizer, max_sequence_len, exact_reimplementation=False):
    """converts a list of Example's into a list of Features"""

    if model_type == "bert":
        model_specs = {"MODEL_TYPE": "bert", "CLS_TOKEN": "[CLS]", "UNK_TOKEN": "[UNK]", "SEP_TOKEN": "[SEP]", "TOKEN_CORRECTION": 4}
    else:
        logger.error("Unknown model type (%s). Aborting." % (model_type))
        exit(1)

    def _get_nonzero_indices(target):
        if type(target) == dict:
            nonzero_indices = {}
            for key in target:
                nonzero_indices[key] = []
                for idx, item in enumerate(target[key]):
                    if item != 0:
                        nonzero_indices[key].append(idx)
        elif type(target) == list:
            nonzero_indices = []
            for idx, item in enumerate(target):
                if item != 0:
                    nonzero_indices.append(idx)
        return nonzero_indices

    def _tokenize_text_and_label(text, text_label, tokenizer):
        tokens = []
        token_labels = []
        for token, label in zip(text, text_label):
            token = convert_to_unicode(token)
            sub_tokens = tokenizer.tokenize(token)
            tokens.extend(sub_tokens)
            token_labels.extend([label for _ in sub_tokens])
        assert len(tokens) == len(token_labels)
        return tokens, token_labels

    def _truncate_sequence(usr_tokens, sys_tokens, hst_tokens, max_sequence_len, guid):
        # Modifies usr, sys, and hst tokens in place to account for special tokens
        # in bert, we use [CLS], [SEP], [SEP], [SEP] for beggining and between token sources
        text_too_long = False
        seq_len = len(usr_tokens) + len(sys_tokens) + len(hst_tokens)
        adjusted_max_seq_len = max_sequence_len - model_specs["TOKEN_CORRECTION"]
        if seq_len > adjusted_max_seq_len:
            # only log the guid as being too long the first time we see it
            if guid not in too_long_guid:
                too_long_guid.append(guid)
                logger.info(f"Truncate Example {guid}. Total len: {len(usr_tokens)+len(sys_tokens)+len(hst_tokens)}")
            text_too_long = True
            # if sequence is too long, remove tokens from the end, prefer to remove from hst, then sys, then usr
            while seq_len > adjusted_max_seq_len:
                if len(hst_tokens) > 0:
                    hst_tokens.pop()
                elif len(sys_tokens) > len(usr_tokens):
                    sys_tokens.pop()
                else:
                    usr_tokens.pop()
                seq_len = len(usr_tokens) + len(sys_tokens) + len(hst_tokens)
        return text_too_long

    def _combine_token_label_ids(usr_token_labels, sys_token_labels, hst_token_labels):
        token_labels = []
        token_labels.append(0)  # [CLS]
        for label in usr_token_labels:
            token_labels.append(label)
        token_labels.append(0)  # [SEP]
        for label in sys_token_labels:
            token_labels.append(label)
        token_labels.append(0)  # [SEP]
        for label in hst_token_labels:
            token_labels.append(label)
        while len(token_labels) < max_sequence_len:
            token_labels.append(0)  # PADDING
        assert len(token_labels) == max_sequence_len
        return token_labels

    def _get_start_end_labels(token_labels):
        start_label = [0 for _ in range(max_sequence_len)]
        end_label = [0 for _ in range(max_sequence_len)]
        if 1 in token_labels:
            # handle first token separately
            if token_labels[0] == 1:
                start_label[0] = 1
                if token_labels[1] == 0:
                    end_label[0] = 1

            for i in range(1, max_sequence_len - 1):
                if token_labels[i] == 1:
                    if token_labels[i - 1] == 0:
                        start_label[i] = 1
                    if token_labels[i + 1] == 0:
                        end_label[i] = 1

            # handle last token separately
            if token_labels[-1] == 1:
                end_label[-1] = 1
                if token_labels[-2] == 0:
                    start_label[-1] = 1

        return start_label, end_label

    def _get_transformer_input(usr_tokens, sys_tokens, hst_tokens):
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append(model_specs["CLS_TOKEN"])
        segment_ids.append(0)
        for token in usr_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(model_specs["SEP_TOKEN"])
        segment_ids.append(0)
        for token in sys_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs["SEP_TOKEN"])
        segment_ids.append(1)
        for token in hst_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs["SEP_TOKEN"])
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # real tokens get a mask value of 1, padding tokens get mask of 0
        attention_mask = [1] * len(input_ids)

        # zero-pad up to max sequence length
        while len(input_ids) < max_sequence_len:
            input_ids.append(0)
            attention_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_sequence_len
        assert len(attention_mask) == max_sequence_len
        assert len(segment_ids) == max_sequence_len
        return tokens, input_ids, attention_mask, segment_ids

    tot = 0
    num_truncated = 0
    features = []
    refer_list = ["none"] + slot_list

    too_long_guid = []

    print("Converting examples to features")
    for idx, example in enumerate(tqdm(examples)):
        if idx % 1000 == 0:
            logger.info(f"Writing example {idx} of {len(examples)}")

        start_labels = {}
        end_labels = {}
        seen_values = {}
        values = {}
        value_sources = {}
        dialog_states = {}
        inform_values = {}
        inform_slot_labels = {}
        refer_labels = {}
        DB_labels = {}
        for slot in slot_list:

            usr_tokens, usr_token_labels = _tokenize_text_and_label(
                example.usr_utterance_tokens, example.usr_utterance_token_label_dict[slot], tokenizer
            )
            sys_tokens, sys_token_labels = _tokenize_text_and_label(
                example.sys_utterance_tokens, example.sys_utterance_token_label_dict[slot], tokenizer
            )
            hst_tokens, hst_token_labels = _tokenize_text_and_label(example.history, example.hst_utterance_token_label_dict[slot], tokenizer)

            text_too_long = _truncate_sequence(usr_tokens, sys_tokens, hst_tokens, max_sequence_len, example.guid)
            if text_too_long:
                if idx < 10:
                    if len(usr_token_labels) > len(usr_tokens):
                        logger.info(f"    usr_tokens truncated from {len(usr_token_labels)} to {len(usr_tokens)}")
                    if len(sys_token_labels) > len(sys_tokens):
                        logger.info(f"    sys_tokens truncated from {len(sys_token_labels)} to {len(sys_tokens)}")
                    if len(usr_token_labels) > len(usr_tokens):
                        logger.info(f"    hst_tokens truncated from {len(hst_token_labels)} to {len(hst_tokens)}")

                usr_token_labels = usr_token_labels[: len(usr_tokens)]
                sys_token_labels = sys_token_labels[: len(sys_tokens)]
                hst_token_labels = hst_token_labels[: len(hst_tokens)]

            assert len(usr_token_labels) == len(usr_tokens)
            assert len(sys_token_labels) == len(sys_tokens)
            assert len(hst_token_labels) == len(hst_tokens)

            token_labels = _combine_token_label_ids(usr_token_labels, sys_token_labels, hst_token_labels)
            start_labels[slot], end_labels[slot] = _get_start_end_labels(token_labels)

            seen_values[slot] = example.seen_values[slot]
            values[slot] = example.values[slot]
            value_sources[slot] = example.value_sources[slot]
            dialog_states[slot] = example.dialog_states[slot]
            inform_values[slot] = example.inform_value[slot]
            inform_slot_labels[slot] = example.inform_slot_label[slot]
            refer_labels[slot] = refer_list.index(example.refer_label[slot])
            DB_labels[slot] = example.DB_label[slot]

        if text_too_long:
            num_truncated += 1
        tot += 1

        tokens, input_ids, attention_mask, segment_ids = _get_transformer_input(usr_tokens, sys_tokens, hst_tokens)

        # for debugging purposes
        if sum(refer_labels.values()) > 0:
            a = 1
        if sum(inform_slot_labels.values()) > 0:
            a = 1

        if idx < 10:
            logger.info("*** Example ***")
            logger.info(f"guid: {example.guid}")
            logger.info(f"tokens: {' '.join(tokens)}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.info(f"attention mask: {' '.join([str(x) for x in attention_mask])}")
            logger.info(f"segment ids: {' '.join([str(x) for x in segment_ids])}")
            logger.info(f"start labels: {_get_nonzero_indices(start_labels)}")
            logger.info(f"end labels: {_get_nonzero_indices(end_labels)}")
            logger.info(f"seen values: {seen_values}")
            logger.info(f"values: {values}")
            logger.info(f"value sources: {value_sources}")
            logger.info(f"dialog state: {dialog_states}")
            logger.info(f"inform labels: {inform_values}")
            logger.info(f"inform slot labels: {inform_slot_labels}")
            logger.info(f"refer labels: {refer_labels}")
            logger.info(f"DB labels: {DB_labels}")

        features.append(
            Features(
                guid=example.guid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                start_labels=start_labels,
                end_labels=end_labels,
                seen_values=seen_values,
                values=values,
                value_sources=value_sources,
                dialog_states=dialog_states,
                inform_values=inform_values,
                inform_slot_labels=inform_slot_labels,
                refer_labels=refer_labels,
                DB_labels=DB_labels,
            )
        )

    logger.info(f"============ TRUNCATED EXAMPLES: {num_truncated} out of {tot}")
    return features


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class TensorListDataset(Dataset):
    """Dataset wrapping tensors, tensor dicts, and tensor lists

    *data (Tensor or dict or list of Tensors): tensors that all have the same size in the first dimension
    """

    def __init__(self, *data):
        if isinstance(data[0], dict):
            size = list(data[0].values())[0].size(0)
        elif isinstance(data[0], list):
            if isinstance(data[0][0], str):
                size = len(data[0])
            else:
                size = data[0][0].size(0)
        else:
            size = data[0].size(0)
        for element in data:
            if isinstance(element, dict):
                if isinstance(list(element.values())[0], list):
                    assert all(size == len(l) for name, l in element.items())  # dict of lists
                else:
                    assert all(size == tensor.size(0) for name, tensor in element.items())  # dict of tensors

            elif isinstance(element, list):
                if isinstance(element[0], str):
                    continue
                assert all(size == tensor.size(0) for tensor in element)  # list of tensors
            else:
                assert size == element.size(0)  # tensor
        self.size = size
        self.data = data

    def __getitem__(self, index):
        result = []
        for element in self.data:
            if isinstance(element, dict):
                result.append({k: v[index] for k, v in element.items()})
            elif isinstance(element, list):
                if isinstance(element[index], str):
                    result.append(element[index])
                else:
                    result.append(v[index] for v in element)
            else:
                result.append(element[index])
        return tuple(result)

    def __len__(self):
        return self.size