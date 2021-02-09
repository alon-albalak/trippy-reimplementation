# This file will contain all functions/variables related to the multiwoz 2.1 dataset
# Including:
#   Loading files (dialog acts, dialogs)
#   Processing files (dialog acts, dialogs) includes:
#       normalizing text (time, common errors),
#       normalizing labels
#       creating turn labels
import os
import json
import re
import logging
from tqdm import tqdm
from data_utils.data_utils import Example, convert_to_unicode

logger = logging.getLogger(__name__)

# Required for mapping slot names in dialogue_acts.json file
# to proper designations.
ACTS_DICT = {
    "taxi-depart": "taxi-departure",
    "taxi-dest": "taxi-destination",
    "taxi-leave": "taxi-leaveAt",
    "taxi-arrive": "taxi-arriveBy",
    "train-depart": "train-departure",
    "train-dest": "train-destination",
    "train-leave": "train-leaveAt",
    "train-arrive": "train-arriveBy",
    "train-people": "train-book_people",
    "restaurant-price": "restaurant-pricerange",
    "restaurant-people": "restaurant-book_people",
    "restaurant-day": "restaurant-book_day",
    "restaurant-time": "restaurant-book_time",
    "hotel-price": "hotel-pricerange",
    "hotel-people": "hotel-book_people",
    "hotel-day": "hotel-book_day",
    "hotel-stay": "hotel-book_stay",
    "booking-people": "booking-book_people",
    "booking-day": "booking-book_day",
    "booking-stay": "booking-book_stay",
    "booking-time": "booking-book_time",
}

# Loads the dialogue_acts.json and returns a list
# of slot-value pairs.
def load_acts(input_file):
    with open(input_file) as f:
        acts = json.load(f)
    s_dict = {}
    for d in acts:
        for t in acts[d]:
            # Only process, if turn has annotation
            if isinstance(acts[d][t], dict):
                for a in acts[d][t]:
                    aa = a.lower().split("-")
                    if aa[1] == "inform" or aa[1] == "recommend" or aa[1] == "select" or aa[1] == "book":
                        for i in acts[d][t][a]:
                            s = i[0].lower()
                            v = i[1].lower().strip()
                            if s == "none" or v == "?" or v == "none":
                                continue
                            slot = aa[0] + "-" + s
                            if slot in ACTS_DICT:
                                slot = ACTS_DICT[slot]
                            key = d + ".json", t, slot
                            # In case of multiple mentioned values...
                            # ... Option 1: Keep first informed value
                            if key not in s_dict:
                                s_dict[key] = list([v])
                            # ... Option 2: Keep last informed value
                            # s_dict[key] = list([v])
    return s_dict


def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text)  # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text)  # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text)  # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text)  # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub(
        "(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text
    )
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour
    return text


def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text)  # Systematic typo
    text = re.sub("guesthouse", "guest house", text)  # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text)  # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text)  # Normalization
    return text


def tokenize(utterance):
    """convert an utterance into tokens, first normalizing values"""
    utterance = convert_to_unicode(utterance).lower()
    utterance = normalize_text(utterance)
    utterance_tokens = [token for token in map(str.strip, re.split("(\W+)", utterance)) if len(token) > 0]
    return utterance_tokens


# This should only contain label normalizations. All other mappings should
# be defined in LABEL_MAPS.
def normalize_label(slot, value_label):
    # Normalization of empty slots
    if value_label == "" or value_label == "not mentioned":
        return "none"

    # Normalization of time slots
    if "leaveAt" in slot or "arriveBy" in slot or slot == "restaurant-book_time":
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)

    # Map to boolean slots
    if slot == "hotel-parking" or slot == "hotel-internet":
        if value_label == "yes" or value_label == "free":
            return "true"
        if value_label == "no":
            return "false"
    if slot == "hotel-type":
        if value_label == "hotel":
            return "true"
        if value_label == "guest house":
            return "false"

    return value_label


def is_in_list(tokens, value):
    found = False
    token_list = [w for w in map(str.strip, re.split("(\W+)", tokens)) if len(w) > 0]
    value_list = [w for w in map(str.strip, re.split("(\W+)", value)) if len(w) > 0]
    t_len = len(token_list)
    v_len = len(value_list)
    for i in range(t_len + 1 - v_len):
        if token_list[i : i + v_len] == value_list:
            found = True
            break
    return found


def check_slot_inform(value, inform_value, value_variations):
    # fuzzy matching for labelling informed slot values
    informed_value = "none"
    result = False
    normalized_val = " ".join(tokenize(value))
    for label in inform_value:
        if normalized_val == label:
            result = True
        elif is_in_list(normalized_val, label):
            result = True
        elif is_in_list(label, normalized_val):
            result = True
        elif label in value_variations:
            for label_variation in value_variations[label]:
                if normalized_val == label_variation:
                    result = True
                    break
                elif is_in_list(normalized_val, label_variation):
                    result = True
                    break
                elif is_in_list(label_variation, normalized_val):
                    result = True
                    break
        elif normalized_val in value_variations:
            for value_variation in value_variations[normalized_val]:
                if value_variation == label:
                    result = True
                    break
                elif is_in_list(value_variation, label):
                    result = True
                    break
                elif is_in_list(label, value_variation):
                    result = True
                    break
        if result:
            informed_value = label
            break
    return result, informed_value


def check_slot_referral(value, slot, seen_slots, value_variations):
    referred_slot = "none"

    # slots that cannot refer to other slots, also cannot be referred to
    non_referrable_slots = ["hotel_stars", "hotel_internet", "hotel_parking"]

    if slot in non_referrable_slots:
        return referred_slot

    for s in seen_slots:
        if s in non_referrable_slots:
            continue
        if re.match("(hotel|restaurant)-book_people", s) and slot == "hotel-book_stay":
            continue
        if re.match("(hotel|restaurant)-book_people", slot) and s == "hotel-book_stay":
            continue
        if slot != s and (slot not in seen_slots or seen_slots[slot] != value):
            # ALON NOTE: they just take the first slot which has the same value
            #   There is no handling of multiple slots which all contain the same value
            if seen_slots[s] == value:
                referred_slot = s
                break
            elif value in value_variations:
                for value_variation in value_variations[value]:
                    if seen_slots[s] == value_variation:
                        referred_slot = s
                        break
    return referred_slot


def get_token_pos(token_list, value):
    # If value exists in token_list, return its position
    find_pos = []
    found = False
    split_value = [w for w in map(str.strip, re.split("(\W+)", value)) if len(w) > 0]
    len_value = len(split_value)
    for i in range(len(token_list) + 1 - len_value):
        if token_list[i : i + len_value] == split_value:
            find_pos.append((i, i + len_value))
            found = True
    return found, find_pos


def check_value_existence(value, utterance_tokens, value_variations):
    # check if value_label exists in utterance
    in_utterance, utterance_pos = get_token_pos(utterance_tokens, value)
    if not in_utterance and value in value_variations:
        for value_variation in value_variations[value]:
            in_utterance, utterance_pos = get_token_pos(utterance_tokens, value_variation)
            if in_utterance:
                break
    return in_utterance, utterance_pos


# slots to possibly exclude from DB:
#   train-arriveBy, other time slots, etc.
#   true/false slots??
# Characters to ignore |><
def check_DB(value_label, slot, value_variations, DB_values):
    # Placeholder, this will return the position of the value label within the DB
    return [0]


# ALON NOTE: keep in mind that this function sometimes accidentally finds overlapping values
#       eg. in MUL0011.json-4, the user asks for a hotel to stay at in the centre.
#       the user previously asked for a restaurant in the centre, and now this function
#           does not differentiate whether centre refers to hotel or restaurant, it applies it to both
def get_turn_sources_and_labels(
    usr_utterance_tokens,
    sys_utterance_tokens,
    value_label,
    inform_value,
    DB_values,
    slot,
    source_dict,
    value_variations,
    seen_slots,
    label_only_last_occurence,
):
    # Takes as input the value label (GT slot value)
    #   determines which sources contain the value
    #   returns a list of the ground truth sources (0 for not contained, 1 if the source contains the value)
    value_sources = [0] * len(source_dict)
    usr_utterance_token_label = [0 for _ in usr_utterance_tokens]
    sys_utterance_token_label = [0 for _ in sys_utterance_tokens]
    informed_value = "none"
    referred_slot = "none"
    DB_label = [0]# temporary since DB is not implemented

    if value_label in ["none", "dontcare", "true", "false"]:
        value_sources[source_dict[value_label]] = 1

    else:
        in_usr, usr_pos = check_value_existence(value_label, usr_utterance_tokens, value_variations)
        if "sys_utt" in source_dict:
            in_sys = False
        else:
            in_sys, sys_pos = check_value_existence(value_label, sys_utterance_tokens, value_variations)
        is_informed, informed_value = check_slot_inform(value_label, inform_value, value_variations)
        referred_slot = check_slot_referral(value_label, slot, seen_slots, value_variations)
        if "DB" in source_dict:
            DB_label = [0]
        else:
            DB_label = check_DB(value_label, slot, value_variations, DB_values)

        if in_usr:
            value_sources[source_dict["usr_utt"]] = 1
            if label_only_last_occurence:
                s, e = usr_pos[-1]
                usr_utterance_token_label[s:e] = [1] * (e - s)
            else:
                for s, e in usr_pos:
                    usr_utterance_token_label[s:e] = [1] * (e - s)

        if in_sys:
            value_sources[source_dict["sys_utt"]] = 1
            if label_only_last_occurence:
                s, e = sys_pos[-1]
                sys_utterance_token_label[s:e] = [1] * (e - s)
            else:
                for s, e in sys_pos:
                    sys_utterance_token_label[s:e] = [1] * (e - s)

        if is_informed:
            value_sources[source_dict["inform"]] = 1

        if referred_slot != "none":
            value_sources[source_dict["refer"]] = 1

    return value_sources, usr_utterance_token_label, sys_utterance_token_label, informed_value, referred_slot, DB_label


def load_multiwoz21_dataset(
    dataset_type="debugging",  # usually train/val/test
    label_value_repetitions=True,
    label_only_last_occurence=True,  # whether we should label all(or only last) occurences of a label in usr and sys utterances
    data_path="data/MULTIWOZ2.1",
    DB_file="",
    sources=["none", "dontcare", "usr_utt", "sys_utt", "inform", "refer", "DB", "true", "false"],
    log_unpointable_values=False,
):

    dataset_file = os.path.join(data_path, f"{dataset_type}_dials.json")
    config_file = os.path.join(data_path, "config.json")
    acts_file = os.path.join(data_path, "dialogue_acts.json")

    check_unpointable = False
    unpointable_in_hst = 0
    unpointable_unknown = 0
    tot_samples = 0

    # source labels - none, dontcare, usr_utt, sys_utt, inform, refer, DB
    source_dict = {label: i for i, label in enumerate(sources)}

    # load the dataset
    with open(dataset_file, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    # load dialog acts file
    system_inform_dict = load_acts(acts_file)
    with open(config_file, "r") as f:
        raw_config = json.load(f)
    slot_list = raw_config["slots"]
    value_variations = raw_config["label_maps"]
    inverse_value_variations = {vv: k for k, v in value_variations.items() for vv in v}

    if os.path.exists(DB_file):
        with open(DB_file) as f:
            DB_values = json.load(f)
    else:
        DB_values = {}

    data = []
    logger.info(f"************* Creating {dataset_type} samples ***************")
    for dialog_id, dialog_data in tqdm(raw_dataset.items()):
        turns = dialog_data["log"]

        cumulative_labels = {slot: "none" for slot in slot_list}

        # utterance token list starts off with an empty entry because system is first in the order, but
        #       the system never has the first utterance
        utterance_token_list = [[]]
        modified_slots_list = [{}]

        # first, collect turn utterances and metadata (labels)
        # usr_sys_switch = True # true when system turn
        turn_itr = 0
        for turn in turns:
            # in multiwoz 2.1, only system turns have metadata
            is_system_utterance = turn["metadata"] != {}

            # TODO: This can probably be removed, I've never seen any issues
            # if usr_sys_switch == is_system_utterance:
            #     print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
            #     break
            # usr_sys_switch = is_system_utterance

            if is_system_utterance:
                turn_itr += 1

            # if we want to delexicalize system utterances, do so here

            # split the current turn utterance into tokens
            utterance_token_list.append(tokenize(turn["text"]))

            modified_slots = {}

            # If this is a system utterance, multiwoz 2.1 has metadata to extract for this turn
            if is_system_utterance:
                for domain in turn["metadata"]:
                    booked = turn["metadata"][domain]["book"]["booked"]
                    booked_slots = {}
                    if booked:
                        for slot in booked[0]:
                            # if we want to adjust the labels from as they appear in the booking metadata, do so here
                            # eg. if we want to convert hotel/guesthouse into true/false, this is a place to do so
                            booked_slots[slot] = normalize_label(f"{domain}-{slot}", booked[0][slot])

                    for category in ["book", "semi"]:
                        for slot in turn["metadata"][domain][category]:
                            ds = f"{domain}-book_{slot}" if category == "book" else f"{domain}-{slot}"
                            value_label = normalize_label(ds, turn["metadata"][domain][category][slot])
                            # ALON NOTE: they prefer the slot-value as stored in the booked section, over the rest
                            if slot in booked_slots:
                                value_label = booked_slots[slot]

                            # track the dialogue state as well as any slots new to this turn
                            if ds in slot_list and cumulative_labels[ds] != value_label:
                                modified_slots[ds] = value_label
                                cumulative_labels[ds] = value_label

            modified_slots_list.append(modified_slots.copy())

        # Form turns
        # For now, track slot values in the same form as TripPy + as DB value
        # First, track utterance tokens
        # Then, for each domain-slot pair:
        #   determine the value
        #   determine from which sources the value can be found
        #   track which tokens contain the value
        #   track the index of the value in the DB (if it exists)
        turn_itr = 0
        sys_utterance_tokens = []
        usr_utterance_tokens = []
        hst_utterance_tokens = []
        hst_utterance_token_label_dict = {slot: [] for slot in slot_list}
        dialog_seen_slots_dict = {}  # dict of {slot: value_source} where we only have slots that have occured in the dialogue and value source
        dialog_seen_slots_value_dict = {slot: "none" for slot in slot_list}
        dialog_state = {slot: [1] + [0] * (len(sources) - 1) for slot in slot_list}
        for i in range(1, len(utterance_token_list) - 1, 2):
            value_sources_dict = {}
            inform_dict = {}
            inform_slot_dict = {}
            referral_dict = {}
            sys_utterance_token_label_dict = {}
            usr_utterance_token_label_dict = {}
            DB_label_dict = {}

            # gather current turn + past turns
            hst_utterance_tokens = usr_utterance_tokens + sys_utterance_tokens + hst_utterance_tokens
            sys_utterance_tokens = utterance_token_list[i - 1]
            usr_utterance_tokens = utterance_token_list[i]
            turn_modified_slots = modified_slots_list[i + 1]

            guid = f"{dataset_type}-{dialog_id}-{turn_itr}"

            new_hst_utterance_token_label_dict = hst_utterance_token_label_dict.copy()
            new_dialog_state = dialog_state.copy()
            new_dialog_seen_slots_value_dict = dialog_seen_slots_value_dict.copy()
            for slot in slot_list:
                # By default the value for each slot is "none"
                value_label = "none"
                # if this slot was modified this turn, update the value label
                if slot in turn_modified_slots:
                    value_label = turn_modified_slots[slot]
                # if this slot was modified in any previous turn, update the value label
                elif label_value_repetitions and slot in dialog_seen_slots_dict:
                    value_label = new_dialog_seen_slots_value_dict[slot]

                # get dialog act annotations
                # inform label is not actually used for classification, just inform_slot_dict
                # however, inform_value should be used to reconcile the labels between all sources
                inform_value = ["none"]
                inform_slot_dict[slot] = 0
                id_itr_slot_tuple = (dialog_id, str(turn_itr), slot)
                id_itr_book_tuple = (dialog_id, str(turn_itr), f"booking-{slot.split('-')[1]}")
                if id_itr_slot_tuple in system_inform_dict:
                    inform_value = [normalize_label(slot, label) for label in system_inform_dict[id_itr_slot_tuple]]
                    inform_slot_dict[slot] = 1
                elif id_itr_book_tuple in system_inform_dict:
                    inform_value = [normalize_label(slot, label) for label in system_inform_dict[id_itr_book_tuple]]
                    inform_slot_dict[slot] = 1

                (
                    value_sources,
                    usr_utterance_token_label,
                    sys_utterance_token_label,
                    informed_value,
                    referred_slot,
                    DB_label,
                ) = get_turn_sources_and_labels(
                    usr_utterance_tokens,
                    sys_utterance_tokens,
                    value_label,
                    inform_value,
                    DB_values,
                    slot,
                    source_dict,
                    value_variations,
                    new_dialog_seen_slots_value_dict,
                    label_only_last_occurence,
                )
                DB_label_dict[slot] = DB_label
                referral_dict[slot] = referred_slot
                inform_dict[slot] = informed_value
                sys_utterance_token_label_dict[slot] = sys_utterance_token_label
                usr_utterance_token_label_dict[slot] = usr_utterance_token_label
                new_hst_utterance_token_label_dict[slot] = (
                    usr_utterance_token_label + sys_utterance_token_label + new_hst_utterance_token_label_dict[slot]
                )

                # in case a value is unpointable, set the value source label to none
                # ALON TODO: Analyze where these unpointable values come from
                #   number of unpointable values reduced from 11% of train data to 1.1% of train data
                # some of these values are listed as multiple possible values in the dataset
                #       eg. "kings college|hughes hall"
                tot_samples += 1
                if sum(value_sources) == 0 and slot in turn_modified_slots:
                    if log_unpointable_values:
                        logger.info(f"Unpointable value {value_label} in {guid} turn {i} slot {slot}")
                    # as a backup, set the source to none, but still add it to the list of values seen
                    value_sources[source_dict["none"]] = 1
                    dialog_seen_slots_dict[slot] = value_sources
                    new_dialog_seen_slots_value_dict[slot] = value_label
                    # check if the value exists in history (but not in current turn)
                    #   ~80% of unpointable values are in the dialogue history
                    in_hst, _ = check_value_existence(value_label, hst_utterance_tokens, value_variations)
                    if in_hst:
                        unpointable_in_hst += 1
                    else:
                        unpointable_unknown += 1
                # in case that the value was previously seen and not repeated in this turn, set the source of the value to "none"
                elif sum(value_sources) == 0 and slot in dialog_seen_slots_dict:
                    value_sources[source_dict["none"]] = 1
                elif sum(value_sources) > 0 and value_sources[source_dict["none"]] == 0:
                    dialog_seen_slots_dict[slot] = value_sources
                    new_dialog_seen_slots_value_dict[slot] = value_label

                elif sum(value_sources) == 1 and value_sources[source_dict["none"]] == 1:
                    pass
                else:
                    logger.info(f"====== Unknown source of value in {guid}\tturn {i}\t{slot}")
                new_dialog_state[slot] = value_sources
                value_sources_dict[slot] = value_sources

            data.append(
                Example(
                    guid=guid,
                    value_sources=value_sources_dict,  # ground truth value sources for this turn
                    usr_utterance_tokens=usr_utterance_tokens,  # usr utterance tokens for this turn
                    sys_utterance_tokens=sys_utterance_tokens,  # sys utterance tokens for this turn
                    history=hst_utterance_tokens,  # history of tokens from all previous turns
                    usr_utterance_token_label_dict=usr_utterance_token_label_dict,  # value labels for current usr utterance tokens
                    sys_utterance_token_label_dict=sys_utterance_token_label_dict,  # value labels for current sys utterance tokens
                    hst_utterance_token_label_dict=hst_utterance_token_label_dict,  # value labels for tokens from all previous turns
                    seen_values=dialog_seen_slots_value_dict.copy(),  # ground truth list of values previously seen (does not include values seen this turn)
                    values=new_dialog_seen_slots_value_dict.copy(),  # ground truth list of values (including this turn and all previous turns)
                    dialog_states=dialog_state,  # ground truth value sources for previous turn
                    inform_value=inform_dict,
                    inform_slot_label=inform_slot_dict,  # ground truth labels for if system informed a value this turn
                    refer_label=referral_dict,  # ground truth labels for which slot is being referred to (can be 'none')
                    DB_label=DB_label_dict,
                )
            )

            # update history with current turn
            dialog_state = new_dialog_state.copy()
            dialog_seen_slots_value_dict = new_dialog_seen_slots_value_dict.copy()
            hst_utterance_token_label_dict = new_hst_utterance_token_label_dict.copy()
            turn_itr += 1

    if check_unpointable:
        logger.info(f"UNPOINTABLE IN HIST: {unpointable_in_hst}\tUNPOINTABLE UNKNOWN SOURCE: {unpointable_unknown}\tTOTAL: {tot_samples}")

    return data