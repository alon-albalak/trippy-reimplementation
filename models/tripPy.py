# This file will contain a pytorch model
from numpy import prod
import re
import torch
from torch.nn import Dropout, Linear, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel


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
        self.downweight_none_slot = config.downweight_none_slot

        slot_weighting = torch.tensor([self.downweight_none_slot] + [1] * (self.num_sources - 1))
        self.source_loss_fct = BCEWithLogitsLoss(weight=slot_weighting, reduction="none")
        self.token_loss_fct = BCEWithLogitsLoss(reduction="none")
        self.refer_loss_fct = CrossEntropyLoss(reduction="none")

        # add module for using inform slots as auxiliary feature
        self.aux_inform_projection = Linear(len(self.slot_list), len(self.slot_list))
        # add module for using dialog state as auxiliary feature
        self.aux_ds_projection = Linear(len(self.slot_list) * self.num_sources, len(self.slot_list))
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
        dialog_state_labels = torch.reshape(torch.stack(list(dialog_states.values()), 1), (inform_values.shape[0], -1))

        pooled_output_aux = torch.cat((pooled_BERT, self.aux_inform_projection(inform_values), self.aux_ds_projection(dialog_state_labels)), 1)

        total_loss = 0
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

            source_loss = torch.sum(self.source_loss_fct(per_slot_source_logits[slot], value_sources[slot]), dim=1)

            start_loss = self.token_loss_fct(per_slot_start_logits[slot], start_labels[slot])
            start_loss = torch.sum(attention_mask * start_loss, dim=1)  # / torch.sum(attention_mask, dim=1)
            end_loss = self.token_loss_fct(per_slot_end_logits[slot], end_labels[slot])
            end_loss = torch.sum(attention_mask * end_loss, dim=1)
            # check if each sample has at least 1 starting token to determine if any tokens are pointable
            token_is_pointable = (torch.sum(start_labels[slot], dim=1) > 0).float()
            token_loss = token_is_pointable * (start_loss + end_loss)

            refer_loss = self.refer_loss_fct(per_slot_refer_logits[slot], refer_labels[slot])
            token_is_referrable = (value_sources[slot][:, self.source_dict["refer"]] == 1).float()
            refer_loss *= token_is_referrable

            per_example_loss = (self.source_loss_ratio * source_loss) + ((1 - self.source_loss_ratio) / 2) * (token_loss + refer_loss)
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
        compute_full_value_distribution,
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

        for slot in self.slot_list:
            if compute_full_value_distribution:
                # predict sources
                pred_source_idx = torch.round(torch.sigmoid(per_slot_source_logits[slot]))
                pred_source = self.sources[pred_source_idx]
                # create an empty list for predicted value token distribution
                value_token_distribution = []
                # for any predicted source, add values to the distribution
                # for any ground truth source, compute accuracy
                if pred_source == "none":
                    pass
                pass

            else:
                # for dialogue state, predict sources (may be multiple), to be passed to next turn?
                pred_sources_for_DS = torch.round(torch.sigmoid(per_slot_source_logits[slot]))
                # pred_value_sources[slot] = pred_sources_for_DS

                # for value prediction, predict a single source
                best_pred_source_idx = torch.argmax(per_slot_source_logits[slot])
                best_pred_source = self.sources[best_pred_source_idx]
                sample_info[f"pred_best_source_{slot}"] = best_pred_source

                pred_start = torch.argmax(per_slot_start_logits[slot])
                pred_end = torch.argmax(per_slot_end_logits[slot])

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
                        pred_dialog_state[slot] = " ".join(input_tokens[pred_start : pred_end + 1])
                        pred_dialog_state[slot] = re.sub("(^| )##", "", pred_dialog_state[slot])
                # ALON TODO: inform_slot_labels only has binary label for if this slot is informed
                #       We need to get the values from someewhere
                elif best_pred_source == "inform":
                    pred_dialog_state[slot] = inform_values[slot][0]

        # to properly use the refer slot, we need a second pass through slots after all others have been filled
        for slot in self.slot_list:
            if compute_full_value_distribution:
                pass
            else:
                best_pred_source_idx = torch.argmax(per_slot_source_logits[slot])
                best_pred_source = self.sources[best_pred_source_idx]
                if best_pred_source == "refer":
                    pred_refer = torch.argmax(per_slot_refer_logits[slot])
                    pred_dialog_state[slot] = seen_values[self.slot_list[int(pred_refer) - 1]][0]

            # add predicted values and ground truth values to sample info
            sample_info[f"pred_value_{slot}"] = pred_dialog_state[slot]
            sample_info[f"ground_truth_value_{slot}"] = values[slot]

            # TODO: check that these are actually correct
            # Compute accuracies
            token_is_pointable = (torch.sum(start_labels[slot], dim=1) > 0).float()
            token_is_referrable = (value_sources[slot][:, self.source_dict["refer"]] == 1).float()
            pred_sources = torch.round(torch.sigmoid(per_slot_source_logits[slot]))
            accuracies["source_acc"][slot] = torch.sum(value_sources[slot] == pred_sources) / prod(pred_sources.shape)

            if token_is_pointable:
                pred_start = torch.round(torch.sigmoid(per_slot_start_logits[slot]))
                pred_end = torch.round(torch.sigmoid(per_slot_end_logits[slot]))
                accuracies["token_acc"][slot] = (
                    torch.sum(token_is_pointable * (start_labels[slot] == pred_start))
                    + torch.sum(token_is_pointable * (end_labels[slot] == pred_end))
                ) / (2 * prod(pred_start.shape))
            else:
                accuracies["token_acc"][slot] = "unpointable"
            if token_is_referrable:
                pred_refer = torch.round(torch.sigmoid(per_slot_refer_logits[slot]))
                accuracies["refer_acc"][slot] = torch.sum(token_is_referrable * (refer_labels[slot] == pred_refer)) / (prod(pred_refer.shape))
            else:
                accuracies["refer_acc"][slot] = "unreferrable"

        return accuracies, pred_dialog_state, sample_info