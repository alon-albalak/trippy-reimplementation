# This file will contain a pytorch model
import logging
from numpy import prod
import re
import torch
from torch.nn import Dropout, Linear, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel
from data_utils.data_utils import decode_normalize_tokens, combine_value_variations_in_value_predictions

logger = logging.getLogger(__name__)


class BERTForDST(BertPreTrainedModel):
    # Flexible BERT model for dialogue state tracking

    def __init__(self, config):
        super(BERTForDST, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = Dropout(config.bert_dropout_rate)
        self.slot_list = config.slot_list
        self.sources = config.sources
        self.source_dict = {source: i for i, source in enumerate(self.sources)}
        self.num_sources = len(self.sources)
        self.source_loss_ratio = config.source_loss_ratio
        if hasattr(config, "exact_reimplementation"):
            self.exact_reimplementation = config.exact_reimplementation
        else:
            self.exact_reimplementation = False

        if self.exact_reimplementation:
            self.source_loss_fct = CrossEntropyLoss(reduction="none")
            self.token_loss_fct = CrossEntropyLoss(reduction="none")
            self.aux_ds_projection = Linear(len(self.slot_list), len(self.slot_list))
        else:
            self.source_loss_fct = BCEWithLogitsLoss(reduction="none")
            self.token_loss_fct = BCEWithLogitsLoss(reduction="none")
            self.aux_ds_projection = Linear(len(self.slot_list) * self.num_sources, len(self.slot_list))

        self.refer_loss_fct = CrossEntropyLoss(reduction="none")

        # add module for using inform slots as auxiliary feature
        self.aux_inform_projection = Linear(len(self.slot_list), len(self.slot_list))
        # add module for using dialog state as auxiliary feature
        aux_dims = 2 * len(self.slot_list)

        for slot in self.slot_list:
            # takes as input BERT embedding + inform_projection + ds_projection, outputs a distribution over sources
            self.add_module(f"source_{slot}", Linear(config.hidden_size + aux_dims, self.num_sources))
            # takes as input a BERT embedding, outputs 2 values (start, end)
            self.add_module(f"token_{slot}", Linear(config.hidden_size, 2))
            # takes as input a BERT embedding + inform_projection + ds_projection, outputs a distribution over ['none']+slots
            self.add_module(f"refer_{slot}", Linear(config.hidden_size + aux_dims, len(self.slot_list) + 1))

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        segment_ids,
        dialog_states,
        inform_slot_labels,
    ):
        outputs = self.bert(input_ids, attention_mask, segment_ids)
        last_hidden_layer_BERT = outputs[0]
        pooled_BERT = outputs[1]

        last_hidden_layer_BERT = self.dropout(last_hidden_layer_BERT)
        pooled_BERT = self.dropout(pooled_BERT)

        inform_values = torch.stack(list(inform_slot_labels.values()), 1)
        if self.exact_reimplementation:
            dialog_state_labels = torch.clamp(torch.stack(list(dialog_states.values()), 1).float(), 0.0, 1.0)
        else:
            dialog_state_labels = torch.reshape(torch.stack(list(dialog_states.values()), 1), (inform_values.shape[0], -1))

        pooled_output_aux = torch.cat((pooled_BERT, self.aux_inform_projection(inform_values), self.aux_ds_projection(dialog_state_labels)), 1)

        per_slot_source_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_refer_logits = {}
        per_slot_DB_logits = {}

        for slot in self.slot_list:
            source_logits = getattr(self, f"source_{slot}")(pooled_output_aux)
            token_logits = getattr(self, f"token_{slot}")(last_hidden_layer_BERT)
            start_logits, end_logits = token_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            refer_logits = getattr(self, f"refer_{slot}")(pooled_output_aux)

            per_slot_source_logits[slot] = source_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits
            per_slot_refer_logits[slot] = refer_logits

        # organize model_outputs in such a way that it is according to loss function?
        # organize it so that loss can easily be handled outside this function
        #       so that we can also run evaluation on this without computing loss

        return (per_slot_source_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits)

    def calculate_loss(
        self,
        attention_mask,
        start_labels,
        end_labels,
        segment_ids,
        value_sources,
        refer_labels,
        DB_labels,
        per_slot_source_logits,
        per_slot_start_logits,
        per_slot_end_logits,
        per_slot_refer_logits,
        calculate_accs=False,
    ):

        total_loss = 0
        per_slot_per_example_loss = {}
        total_source_loss = 0
        total_token_loss = 0
        total_refer_loss = 0

        source_acc = 0
        start_acc = 0
        end_acc = 0
        refer_acc = 0
        count = 0

        for slot in self.slot_list:

            if self.exact_reimplementation:
                source_loss = self.source_loss_fct(per_slot_source_logits[slot], value_sources[slot])
                ignored_index = per_slot_start_logits[slot].size(1)
                start_labels[slot].clamp(0, ignored_index)
                end_labels[slot].clamp(0, ignored_index)
                self.token_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=ignored_index)
                start_loss = self.token_loss_fct(per_slot_start_logits[slot], start_labels[slot])
                end_loss = self.token_loss_fct(per_slot_end_logits[slot], end_labels[slot])
                token_is_pointable = (start_labels[slot] > 0).float()
            else:
                source_loss = torch.sum(self.source_loss_fct(per_slot_source_logits[slot], value_sources[slot]), dim=1)
                start_loss = self.token_loss_fct(per_slot_start_logits[slot], start_labels[slot])
                # ALON TODO: check out whether this attention mask is really what we want
                start_loss = torch.sum(attention_mask * start_loss, dim=1)
                end_loss = self.token_loss_fct(per_slot_end_logits[slot], end_labels[slot])
                end_loss = torch.sum(attention_mask * end_loss, dim=1)
                # check if each sample has at least 1 starting token to determine if any tokens are pointable
                token_is_pointable = (torch.sum(start_labels[slot], dim=1) > 0).float()
            token_loss = token_is_pointable * (start_loss + end_loss)

            refer_loss = self.refer_loss_fct(per_slot_refer_logits[slot], refer_labels[slot])

            if self.exact_reimplementation:
                token_is_referrable = (value_sources[slot] == self.source_dict["refer"]).float()
            else:
                token_is_referrable = (value_sources[slot][:, self.source_dict["refer"]] == 1).float()

            refer_loss *= token_is_referrable

            per_example_loss = (self.source_loss_ratio * source_loss) + ((1 - self.source_loss_ratio) / 2) * (token_loss + refer_loss)
            # per_example_loss = source_loss + (0.7 * token_loss) + (0.5 * refer_loss)
            # per_example_loss = source_loss + token_loss + refer_loss
            # ALON NOTE: EQUALLOSSES TRAINING

            total_loss += torch.sum(per_example_loss)
            per_slot_per_example_loss[slot] = per_example_loss
            total_source_loss += torch.sum(source_loss)
            total_token_loss += torch.sum(token_loss)
            total_refer_loss += torch.sum(refer_loss)

            if calculate_accs:
                # Out of curiosity
                pred_sources = torch.round(torch.sigmoid(per_slot_source_logits[slot]))
                source_acc += torch.sum(value_sources[slot] == pred_sources) / prod(pred_sources.shape)
                pred_start = torch.round(torch.sigmoid(per_slot_start_logits[slot]))
                start_acc += torch.sum(token_is_pointable * torch.sum(start_labels[slot] == pred_start, dim=1)) / prod(pred_start.shape)
                pred_end = torch.round(torch.sigmoid(per_slot_end_logits[slot]))
                end_acc += torch.sum(token_is_pointable * torch.sum(end_labels[slot] == pred_end, dim=1)) / prod(pred_end.shape)
                pred_refer = torch.argmax(per_slot_refer_logits[slot], dim=1)
                refer_acc += torch.sum(token_is_referrable * (refer_labels[slot] == pred_refer)) / prod(pred_refer.shape)
                count += 1
        if calculate_accs:
            source_acc = source_acc / count
            start_acc = start_acc / count
            end_acc = end_acc / count
            refer_acc = refer_acc / count
        return (
            total_loss,
            total_source_loss,
            total_token_loss,
            total_refer_loss,
            source_acc,
            start_acc,
            end_acc,
            refer_acc,
        )

    def evaluate(
        self,
        tokenizer,
        guid,
        input_ids,
        attention_mask,
        segment_ids,
        start_labels,
        end_labels,
        seen_values,  # ground truth list of values (not including current turn)
        values,  # ground truth list of values (including this and all previous turns)
        value_sources,  # ground truth dialog state of the current turn (includes all previous turns)
        dialog_state,  # ground truth dialog state from previous turn
        pred_dialog_state,  # predicted dialog state from previous turn
        inform_values,  # dict of {slot: ground truth value} for values which were informed by the system
        inform_slot_labels,  # dict of {slot: 0/1} where the label is 1 if the slot was informed by the system
        refer_labels,
        DB_labels,
        softgate,
        value_variations,
        inverse_value_variations,
    ):
        """Method which calculates statistics for evaluation
        such as source accuracy, token accuracy, refer accuracy
        slot accuracy
        """

        # if using a soft gating mechanism, pass the calculation off to the another function
        if softgate:
            accuracies, pred_dialog_state, sample_info = self.evaluate_softgate(
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
                dialog_state,
                pred_dialog_state,
                inform_values,
                inform_slot_labels,
                refer_labels,
                DB_labels,
                value_variations,
                inverse_value_variations,
            )

        else:
            accuracies = {
                "source_acc": {slot: 0 for slot in self.slot_list},
                "token_acc": {slot: 0 for slot in self.slot_list},
                "refer_acc": {slot: 0 for slot in self.slot_list},
            }
            sample_info = {"guid": guid}

            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                dialog_states=dialog_state,
                inform_slot_labels=inform_slot_labels,
            )
            per_slot_source_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits = outputs

            for slot in self.slot_list:
                # for dialogue state, predict sources (may be multiple), to be passed to next turn?
                pred_sources_for_DS = torch.round(torch.sigmoid(per_slot_source_logits[slot]))
                # pred_value_sources[slot] = pred_sources_for_DS

                # for value prediction, predict a single source
                best_pred_source_idx = torch.argmax(per_slot_source_logits[slot])
                best_pred_source = self.sources[best_pred_source_idx]
                sample_info[f"pred_sources_{slot}"] = best_pred_source

                pred_start = torch.argmax(per_slot_start_logits[slot])
                pred_end = torch.argmax(per_slot_end_logits[slot])

                if self.exact_reimplementation:
                    ground_truth_sources = [self.sources[value_sources[slot]]]
                else:
                    ground_truth_source_idxs = torch.nonzero(value_sources[slot].squeeze() == 1)
                    ground_truth_sources = []
                    for idx in ground_truth_source_idxs:
                        ground_truth_sources.append(self.sources[idx])
                sample_info[f"ground_truth_sources_{slot}"] = ground_truth_sources
                ground_truth_start_idxs = torch.nonzero(start_labels[slot].squeeze() == 1)
                ground_truth_end_idxs = torch.nonzero(end_labels[slot].squeeze() == 1)
                ground_truth_refer = refer_labels[slot]

                # sample_info["source_pred"] = pred_source
                # sample_info["source_GT"] = ground_truth_sources
                # sample_info["start_pred"] = pred_start
                # sample_info["start_GT"] = ground_truth_start_idxs
                # sample_info["end_pred"] = pred_end
                # sample_info["end_GT"] = ground_truth_end_idxs
                # sample_info['refer_pred'] = ground_truth_refer

                # for any predicted source, add values to the distribution
                # for any ground truth source, compute accuracy
                if best_pred_source in ["none"]:
                    # if predict none, then don't alter pred_dialog_state
                    pass
                elif best_pred_source in ["dontcare", "true", "false"]:
                    pred_dialog_state[slot] = best_pred_source
                elif best_pred_source in ["usr_utt", "sys_utt"]:
                    # if pred_start > pred_end, treat it as "none"
                    if pred_start <= pred_end:
                        input_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
                        pred_dialog_state[slot] = decode_normalize_tokens(input_tokens, pred_start, pred_end)

                elif best_pred_source == "inform":
                    pred_dialog_state[slot] = inform_values[slot][0]

            # to properly use the refer slot, we need a second pass through slots after all others have been filled
            for slot in self.slot_list:

                best_pred_source_idx = torch.argmax(per_slot_source_logits[slot])
                best_pred_source = self.sources[best_pred_source_idx]
                if best_pred_source == "refer":
                    pred_refer = torch.argmax(per_slot_refer_logits[slot])
                    pred_dialog_state[slot] = pred_dialog_state[self.slot_list[int(pred_refer) - 1]][0]

                # add predicted values and ground truth values to sample info
                sample_info[f"pred_value_{slot}"] = pred_dialog_state[slot]
                sample_info[f"ground_truth_value_{slot}"] = values[slot]

                # TODO: check that these are actually correct
                # Compute accuracies
                # token_is_pointable = (torch.sum(start_labels[slot], dim=1) > 0).float()
                # token_is_referrable = (value_sources[slot][:, self.source_dict["refer"]] == 1).float()
                # pred_sources = torch.round(torch.sigmoid(per_slot_source_logits[slot]))
                # accuracies["source_acc"][slot] = torch.sum(value_sources[slot] == pred_sources) / prod(pred_sources.shape)

                # if token_is_pointable:
                #     pred_start = torch.round(torch.sigmoid(per_slot_start_logits[slot]))
                #     pred_end = torch.round(torch.sigmoid(per_slot_end_logits[slot]))
                #     accuracies["token_acc"][slot] = (
                #         torch.sum(token_is_pointable * (start_labels[slot] == pred_start))
                #         + torch.sum(token_is_pointable * (end_labels[slot] == pred_end))
                #     ) / (2 * prod(pred_start.shape))
                # else:
                #     accuracies["token_acc"][slot] = "unpointable"
                # if token_is_referrable:
                #     pred_refer = torch.round(torch.sigmoid(per_slot_refer_logits[slot]))
                #     accuracies["refer_acc"][slot] = torch.sum(token_is_referrable * (refer_labels[slot] == pred_refer)) / (prod(pred_refer.shape))
                # else:
                #     accuracies["refer_acc"][slot] = "unreferrable"

        return accuracies, pred_dialog_state, sample_info

    def calculate_source_values(
        self,
        source,
        source_weight,
        value_prediction_distribution,
        logits=None,
        topk=None,
        tokenizer=None,
        input_ids=None,
        inform_values=None,
        refer_preds=None,
        pred_dialog_state=None,
    ):
        # add binary sources
        if source in ["none", "dontcare", "true", "false"]:
            if source not in value_prediction_distribution:
                value_prediction_distribution[source] = 0
            value_prediction_distribution[source] += source_weight

        # ALON NOTE: This is a pretty significant issue
        # because we calculate token probabilities over the whole context, we can't really split usr vs sys vs previous turn utterances
        elif source in ["usr_utt", "sys_utt"]:
            utterance_value_distribution = {}
            start_logits, start_idxs = torch.topk(logits[0].squeeze(), k=topk)
            end_logits, end_idxs = torch.topk(logits[1].squeeze(), k=topk)
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

            # calculate distribution over all k^2 combinations of start and end tokens
            for start_logit, start_idx in zip(start_logits, start_idxs):
                for end_logit, end_idx in zip(end_logits, end_idxs):
                    # ignore pairs where the predicted start is after the end
                    if start_idx <= end_idx and end_idx - start_idx <= 10:
                        pred_value = decode_normalize_tokens(input_tokens, start_idx, end_idx)
                        if pred_value not in utterance_value_distribution:
                            utterance_value_distribution[pred_value] = 0
                        utterance_value_distribution[pred_value] += start_logit + end_logit

            # convert utterance_value_distribution into tensor to compute softmax
            values = []
            logits = []
            for val, logit in utterance_value_distribution.items():
                values.append(val)
                logits.append(logit)
            logits = torch.tensor(logits)
            value_weights = torch.nn.functional.softmax(logits, dim=0)
            for val, val_weight in zip(values, value_weights):
                # if val_weight > gate_probability_lower_bound:
                if val not in value_prediction_distribution:
                    value_prediction_distribution[val] = 0
                value_prediction_distribution[val] += source_weight * val_weight

        elif source == "inform":
            value = inform_values[0]
            if value not in value_prediction_distribution:
                value_prediction_distribution[value] = 0
            value_prediction_distribution[value] += source_weight

        # ALON TODO: Not totally sure how to handle this
        #       because we normally would wait until a full pass through the predictions to decide what value belongs to the refer
        #
        #       Is it possible to include all referred slots, and use them as a placeholder?
        #       Can we simply reduce this case to binary? If source_weight > 0.5, just go with the refer value_prediction_distribution

        #   BEST OPTION:
        #       Maybe we temporarily leave this one, flag the dict somehow, and then only finish computing total value in next pass
        #           This is only possible if we never have a dual slot referral( slot a refers to slot b, and slot b refers to slot c)
        #           Perhaps in this referception, we simply leave a pointer to which slot we refer to, and then calculate them in the appropriate order later
        # final idea: save a dict of slots which refer to other slots (like pointers)
        #       after first pass through this go through dict, and calculate values
        elif source == "refer":
            for slot, refer_weight in refer_preds.items():
                if slot == "none":
                    referred_value = "none"
                else:
                    referred_value = pred_dialog_state[slot]
                if referred_value not in value_prediction_distribution:
                    value_prediction_distribution[referred_value] = 0
                value_prediction_distribution[referred_value] += source_weight * refer_weight

        return value_prediction_distribution

    def evaluate_softgate(
        self,
        tokenizer,
        guid,
        input_ids,
        attention_mask,
        segment_ids,
        start_labels,
        end_labels,
        seen_values,  # ground truth list of values (not including current turn)
        values,  # ground truth list of values (including this and all previous turns)
        value_sources,  # ground truth dialog state of the current turn (includes all previous turns)
        dialog_state,  # ground truth dialog state from previous turn
        pred_dialog_state,  # predicted dialog state from previous turn
        inform_values,  # dict of {slot: ground truth value} for values which were informed by the system
        inform_slot_labels,  # dict of {slot: 0/1} where the label is 1 if the slot was informed by the system
        refer_labels,
        DB_labels,
        value_variations,
        inverse_value_variations,
        source_topk=3,
        token_topk=3,
        refer_topk=2,
        DB_topk=3,
        gate_probability_lower_bound=1e-2,
    ):
        """Method which calculates statistics for evaluation
        such as source accuracy, token accuracy, refer accuracy
        slot accuracy
        """

        accuracies = {
            "source_acc": {slot: 0 for slot in self.slot_list},
            "token_acc": {slot: 0 for slot in self.slot_list},
            "refer_acc": {slot: 0 for slot in self.slot_list},
        }
        sample_info = {"guid": guid}

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            dialog_states=dialog_state,
            inform_slot_labels=inform_slot_labels,
        )
        per_slot_source_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits = outputs

        # ALON TODO: QUESTIONS:

        #   Question, how do we handle refer?
        #       Calculate all values first, then come back after for a second pass including the refer source?

        refer_list = ["none"] + self.slot_list
        referred_slots = {}

        #   Should we use topk for sources as well as for values? Since we assume that a single value probably doesn't come from > 3 sources. EVER.

        for slot in self.slot_list:
            # ALON TODO Per slot outline:
            #   1. Get ground truth source and value
            ground_truth_source_idxs = torch.nonzero(value_sources[slot].squeeze() == 1)
            ground_truth_sources = []
            for idx in ground_truth_source_idxs:
                ground_truth_sources.append(self.sources[idx])
            sample_info[f"ground_truth_sources_{slot}"] = ground_truth_sources
            sample_info[f"ground_truth_value_{slot}"] = values[slot]

            #   2. Get predicted probability distribution over sources

            # if only selecting top k sources, get their logits and compute softmax
            if source_topk:
                # first get top sources, then compute softmax on remaining sources
                source_logits, source_idxs = torch.topk(per_slot_source_logits[slot].squeeze(), k=source_topk)
                source_weights = torch.nn.functional.softmax(source_logits, dim=0)
                pred_sources = [
                    (self.sources[idx], weight) for idx, weight in zip(source_idxs, source_weights) if weight > gate_probability_lower_bound
                ]
                a = 1
            else:
                source_weights = torch.nn.functional.softmax(per_slot_source_logits[slot].squeeze(), dim=0)
                pred_sources = [(self.sources[idx], weight) for idx, weight in enumerate(source_weights) if weight > gate_probability_lower_bound]
                a = 1

            #   2. For each source - get probability distribution of each value
            #           2 options, if using topk, the best use of softmax would be to get topk on logits, and then softmax over those top k values

            if "refer" in [s[0] for s in pred_sources]:
                # calculate which slot this refers to, save in dict, along with pred sources, weights
                refer_logits, refer_idxs = torch.topk(per_slot_refer_logits[slot].squeeze(), k=refer_topk)
                refer_weights = torch.nn.functional.softmax(refer_logits, dim=0)
                refer_preds = {refer_list[idx]: weight for idx, weight in zip(refer_idxs, refer_weights)}
                referred_slots[slot] = {"refer_preds": refer_preds, "pred_sources": {source: source_weight for source, source_weight in pred_sources}}
                a = 1
                continue

            value_prediction_distribution = {}

            for source, source_weight in pred_sources:
                kwargs = {}
                if source in ["usr_utt", "sys_utt"]:
                    kwargs = {
                        "input_ids": input_ids,
                        "logits": [per_slot_start_logits[slot], per_slot_end_logits[slot]],
                        "tokenizer": tokenizer,
                        "topk": token_topk,
                    }
                elif source == "inform":
                    kwargs = {"inform_values": inform_values[slot]}

                value_prediction_distribution = self.calculate_source_values(source, source_weight, value_prediction_distribution, **kwargs)

            value_prediction_distribution = combine_value_variations_in_value_predictions(
                value_prediction_distribution, value_variations, inverse_value_variations
            )
            # ALON TODO: Add a step to combine value variations?
            #   3. Combine probability distributions
            #   actually, just get the value with highest probability
            highest_prob = 0
            pred_val = "none"
            if not value_prediction_distribution:
                logger.info(f"NO VALUE PREDICTIONS: guid - {guid} - {slot} - pred sources - {pred_sources}")
            for val, prob in value_prediction_distribution.items():
                if prob > highest_prob:
                    pred_val = val
                    highest_prob = prob

            if pred_val != "none":
                pred_dialog_state[slot] = pred_val
            sample_info[f"pred_value_{slot}"] = pred_dialog_state[slot]
            sample_info[f"pred_sources_{slot}"] = [pred[0] for pred in pred_sources]

        # return for the slots which have referral source
        # Although conceptually impossible, it is possible that annotations suggest that 2 slots refer to each other, if this is the case, break the tie arbitrarily
        while referred_slots:
            used_slots = []
            for slot in referred_slots:
                if any(referred_slots[slot]["refer_preds"]) in referred_slots:
                    continue
                value_prediction_distribution = {}
                for source, source_weight in referred_slots[slot]["pred_sources"].items():
                    kwargs = {}
                    if source in ["usr_utt", "sys_utt"]:
                        kwargs = {
                            "input_ids": input_ids,
                            "logits": [per_slot_start_logits[slot], per_slot_end_logits[slot]],
                            "tokenizer": tokenizer,
                            "topk": token_topk,
                        }
                    elif source == "inform":
                        kwargs = {"inform_values": inform_values[slot]}

                    elif source == "refer":
                        kwargs = {"refer_preds": referred_slots[slot]["refer_preds"], "pred_dialog_state": pred_dialog_state}

                    value_prediction_distribution = self.calculate_source_values(source, source_weight, value_prediction_distribution, **kwargs)

                value_prediction_distribution = combine_value_variations_in_value_predictions(
                    value_prediction_distribution, value_variations, inverse_value_variations
                )
                # Combine probability distributions for this slot
                highest_prob = 0
                for val, prob in value_prediction_distribution.items():
                    if prob > highest_prob:
                        pred_val = val
                        highest_prob = prob

                if pred_val != "none":
                    pred_dialog_state[slot] = pred_val
                sample_info[f"pred_value_{slot}"] = pred_dialog_state[slot]
                sample_info[f"pred_sources_{slot}"] = [pred for pred in referred_slots[slot]["pred_sources"]]

                used_slots.append(slot)
            for slot in used_slots:
                del referred_slots[slot]

        return accuracies, pred_dialog_state, sample_info