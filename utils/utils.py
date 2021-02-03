# This file will contain the argument parsing, as well as some global variables
import argparse
import json
from torch import cuda
from transformers import BertConfig, BertTokenizer
from models.tripPy import BERTForDST

MODEL_CLASSES = {"bert": (BertConfig, BERTForDST, BertTokenizer)}
sources = ["none", "dontcare", "usr_utt", "sys_utt", "inform", "refer", "DB", "true", "false"]

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


def parse_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--dataset", type=str, default="multiwoz21")
    parser.add_argument("--do_lower_case", type=bool, default=True)
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--log_unpointable_values", action="store_true")

    # model args
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_sequence_len", type=int, default=180)
    parser.add_argument("--bert_dropout_rate", type=float, default=0.3, help="Dropout rate for BERT representations")

    # training args
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("--effective_batch_size", type=int, default=70)
    parser.add_argument("--gpu_batch_size", type=int, default=35)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--source_loss_ratio", type=float, default=0.8)
    parser.add_argument("--eval_during_training", type=bool, default=True)
    parser.add_argument("--save_model_checkpoints", type=bool, default=True)
    parser.add_argument("--calculate_accs", action="store_true")
    parser.add_argument("--downweight_none_slot", type=float, default=1.0)

    # testing args
    parser.add_argument("--eval_name", type=str, help="name of model to be evaluated")
    parser.add_argument(
        "--softgate",
        action="store_true",
        help="If set to true, during evaluation the model will compute a distribution over all values \
                            from all sources, otherwise the model computes values only over the source with highest probability",
    )
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--test", action="store_true")

    # data args
    parser.add_argument("--data_path", type=str, default="data/MULTIWOZ2.1")
    parser.add_argument("--cache_features", type=bool, default=True)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--DB_file", type=str, default="")
    parser.add_argument("--label_value_repetitions", type=bool, default=True)
    parser.add_argument("--label_only_last_occurence", action="store_true")
    parser.add_argument("--no_sys_utt", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")

    # debugging args
    parser.add_argument("--debugging", action="store_true")

    args = parser.parse_args()

    if not args.cpu_only:
        setattr(args, "device", "cuda" if cuda.is_available() else "cpu")
    else:
        setattr(args, "device", "cpu")

    setattr(args, "sources", sources)
    setattr(args, "slot_list", slot_list)

    return vars(args)