#!/usr/bin/env python
# coding: utf-8
# Windows
# import json
# import os
# from tqdm import tqdm
# import re
# from IPython.core.debugger import set_trace
# from pprint import pprint
# from transformers import AutoModel, BertTokenizerFast
# import copy
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import torch.optim as optim
# import glob
# import time
# import logging
# from m_kg_tplinker_original.TPlinker_joint_extraction.common.utils import Preprocessor, DefaultLogger
# from m_kg_tplinker_original.TPlinker_joint_extraction.tplinker.tplinker import (HandshakingTaggingScheme,
#                                                                                 DataMaker4Bert,
#                                                                                 DataMaker4BiLSTM,
#                                                                                 TPLinkerBert,
#                                                                                 TPLinkerBiLSTM,
#                                                                                 MetricsCalculator)
# import wandb
# from m_kg_tplinker_original.TPlinker_joint_extraction.tplinker import config
# # from glove import Glove
# import numpy as np

# Linux
import json
import os
from tqdm import tqdm
import re
from pprint import pprint
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import time

# from common.utils import Preprocessor, DefaultLogger
import copy


class DefaultLogger:
    def __init__(self, log_path, project, run_name, run_id, hyperparameter):
        self.log_path = log_path
        log_dir = "/".join(self.log_path.split("/")[:-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.run_id = run_id
        self.log("============================================================================")
        self.log("project: {}, run_name: {}, run_id: {}\n".format(project, run_name, run_id))
        hyperparameters_format = "--------------hypter_parameters------------------- \n{}\n-----------------------------------------"
        self.log(hyperparameters_format.format(json.dumps(hyperparameter, indent=4)))

    def log(self, text):
        text = "run_id: {}, {}".format(self.run_id, text)
        print(text)
        open(self.log_path, "a", encoding="utf-8").write("{}\n".format(text))


class Preprocessor:
    '''
    1. transform the dataset to normal format, which can fit in our codes
    2. add token level span to all entities in the relations, which will be used in tagging phase
    '''

    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func

    def transform_data(self, data, ori_format, dataset_type, add_id=True):
        '''
        This function can only deal with three original format used in the previous works.
        If you want to feed new dataset to the model, just define your own function to transform data.
        data: original data
        ori_format: "casrel", "joint_re", "raw_nyt"
        dataset_type: "train", "valid", "test"; only for generate id for the data
        '''
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc="Transforming data format"):
            if ori_format == "casrel":
                text = sample["text"]
                rel_list = sample["triple_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "etl_span":
                text = " ".join(sample["tokens"])
                rel_list = sample["spo_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "raw_nyt":
                text = sample["sentText"]
                rel_list = sample["relationMentions"]
                subj_key, pred_key, obj_key = "em1Text", "label", "em2Text"

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            normal_rel_list = []
            for rel in rel_list:
                normal_rel = {
                    "subject": rel[subj_key],
                    "predicate": rel[pred_key],
                    "object": rel[obj_key],
                }
                normal_rel_list.append(normal_rel)
            normal_sample["relation_list"] = normal_rel_list
            normal_sample_list.append(normal_sample)

        return self._clean_sp_char(normal_sample_list)

    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len=50, encoder="BERT", data_type="train"):
        new_sample_list = []
        for sample in tqdm(sample_list, desc="Splitting into subtexts"):
            text_id = sample["id"]
            text = sample["text"]
            tokens = self._tokenize(text)
            tok2char_span = self._get_tok2char_span_map(text)

            # sliding at token level
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT":  # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]
                sub_text = text[char_level_span[0]:char_level_span[1]]

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "tok_offset": start_ind,
                    "char_offset": char_level_span[0],
                }
                if data_type == "test":  # test set
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else:  # train or valid dataset, only save spo and entities in the subtext
                    # spo
                    sub_rel_list = []
                    for rel in sample["relation_list"]:
                        subj_tok_span = rel["subj_tok_span"]
                        obj_tok_span = rel["obj_tok_span"]
                        # if subject and object are both in this subtext, add this spo to new sample
                        if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                                and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind:
                            new_rel = copy.deepcopy(rel)
                            new_rel["subj_tok_span"] = [subj_tok_span[0] - start_ind, subj_tok_span[1] - start_ind]  # start_ind: tok level offset
                            new_rel["obj_tok_span"] = [obj_tok_span[0] - start_ind, obj_tok_span[1] - start_ind]
                            new_rel["subj_char_span"][0] -= char_level_span[0]  # char level offset
                            new_rel["subj_char_span"][1] -= char_level_span[0]
                            new_rel["obj_char_span"][0] -= char_level_span[0]
                            new_rel["obj_char_span"][1] -= char_level_span[0]
                            sub_rel_list.append(new_rel)

                    # entity
                    sub_ent_list = []
                    for ent in sample["entity_list"]:
                        tok_span = ent["tok_span"]
                        # if entity in this subtext, add the entity to new sample
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                            new_ent = copy.deepcopy(ent)
                            new_ent["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]

                            new_ent["char_span"][0] -= char_level_span[0]
                            new_ent["char_span"][1] -= char_level_span[0]

                            sub_ent_list.append(new_ent)

                    # event
                    if "event_list" in sample:
                        sub_event_list = []
                        for event in sample["event_list"]:
                            trigger_tok_span = event["trigger_tok_span"]
                            if trigger_tok_span[1] > end_ind or trigger_tok_span[0] < start_ind:
                                continue
                            new_event = copy.deepcopy(event)
                            new_arg_list = []
                            for arg in new_event["argument_list"]:
                                if arg["tok_span"][0] >= start_ind and arg["tok_span"][1] <= end_ind:
                                    new_arg_list.append(arg)
                            new_event["argument_list"] = new_arg_list
                            sub_event_list.append(new_event)
                        new_sample["event_list"] = sub_event_list  # maybe empty

                    new_sample["entity_list"] = sub_ent_list  # maybe empty
                    new_sample["relation_list"] = sub_rel_list  # maybe empty
                    split_sample_list.append(new_sample)

                # all segments covered, no need to continue
                if end_ind > len(tokens):
                    break

            new_sample_list.extend(split_sample_list)
        return new_sample_list

    def _clean_sp_char(self, dataset):
        def clean_text(text):
            text = re.sub("�", "", text)
            #             text = re.sub("([A-Za-z]+)", r" \1 ", text)
            #             text = re.sub("(\d+)", r" \1 ", text)
            #             text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(dataset, desc="Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return dataset

    def clean_data_wo_span(self, ori_data, separate=False, data_type="train"):
        '''
        rm duplicate whitespaces
        and add whitespaces around tokens to keep special characters from them
        '''

        def clean_text(text):
            text = re.sub("\s+", " ", text).strip()
            if separate:
                text = re.sub("([^A-Za-z0-9])", r" \1 ", text)
                text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(ori_data, desc="clean data"):
            sample["text"] = clean_text(sample["text"])
            if data_type == "test":
                continue
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return ori_data

    def clean_data_w_span(self, ori_data):
        '''
        stripe whitespaces and change spans
        add a stake to bad samples(char span error) and remove them from the clean data
        '''
        bad_samples, clean_data = [], []

        def strip_white(entity, entity_char_span):
            p = 0
            while entity[p] == " ":
                entity_char_span[0] += 1
                p += 1

            p = len(entity) - 1
            while entity[p] == " ":
                entity_char_span[1] -= 1
                p -= 1
            return entity.strip(), entity_char_span

        for sample in tqdm(ori_data, desc="clean data w char spans"):
            text = sample["text"]

            bad = False
            for rel in sample["relation_list"]:
                # rm whitespaces
                rel["subject"], rel["subj_char_span"] = strip_white(rel["subject"], rel["subj_char_span"])
                rel["object"], rel["obj_char_span"] = strip_white(rel["object"], rel["obj_char_span"])

                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                if rel["subject"] not in text or rel["subject"] != text[subj_char_span[0]:subj_char_span[1]] or \
                        rel["object"] not in text or rel["object"] != text[obj_char_span[0]:obj_char_span[1]]:
                    rel["stake"] = 0
                    bad = True

            if bad:
                bad_samples.append(copy.deepcopy(sample))

            new_rel_list = [rel for rel in sample["relation_list"] if "stake" not in rel]
            if len(new_rel_list) > 0:
                sample["relation_list"] = new_rel_list
                clean_data.append(sample)
        return clean_data, bad_samples

    def _get_char2tok_span(self, text):
        '''
        map character index to token level span
        '''
        tok2char_span = self._get_tok2char_span_map(text)
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1] for _ in range(char_num)]  # [-1, -1] is whitespace
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1
        return char2tok_span

    def _get_ent2char_spans(self, text, entities, ignore_subword_match=True):
        '''
        if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
        '''
        entities = sorted(entities, key=lambda x: len(x), reverse=True)
        text_cp = " {} ".format(text) if ignore_subword_match else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword_match else ent
            for m in re.finditer(re.escape(target_ent), text_cp):
                if not ignore_subword_match and re.match("\d+", target_ent):  # avoid matching a inner number of a number
                    if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or (m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                        continue
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
                spans.append(span)
            #             if len(spans) == 0:
            #                 set_trace()
            ent2char_spans[ent] = spans
        return ent2char_spans

    def add_char_span(self, dataset, ignore_subword_match=True):
        miss_sample_list = []
        for sample in tqdm(dataset, desc="adding char level spans"):
            entities = [rel["subject"] for rel in sample["relation_list"]]
            entities.extend([rel["object"] for rel in sample["relation_list"]])
            if "entity_list" in sample:
                entities.extend([ent["text"] for ent in sample["entity_list"]])
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities, ignore_subword_match=ignore_subword_match)

            new_relation_list = []
            for rel in sample["relation_list"]:
                subj_char_spans = ent2char_spans[rel["subject"]]
                obj_char_spans = ent2char_spans[rel["object"]]
                for subj_sp in subj_char_spans:
                    for obj_sp in obj_char_spans:
                        new_relation_list.append({
                            "subject": rel["subject"],
                            "object": rel["object"],
                            "subj_char_span": subj_sp,
                            "obj_char_span": obj_sp,
                            "predicate": rel["predicate"],
                        })

            if len(sample["relation_list"]) > len(new_relation_list):
                miss_sample_list.append(sample)
            sample["relation_list"] = new_relation_list

            if "entity_list" in sample:
                new_ent_list = []
                for ent in sample["entity_list"]:
                    for char_sp in ent2char_spans[ent["text"]]:
                        new_ent_list.append({
                            "text": ent["text"],
                            "type": ent["type"],
                            "char_span": char_sp,
                        })
                sample["entity_list"] = new_ent_list
        return dataset, miss_sample_list

    def add_tok_span(self, dataset):
        '''
        dataset must has char level span
        '''

        def char_span2tok_span(char_span, char2tok_span):
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span

        for sample in tqdm(dataset, desc="adding token level spans"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for rel in sample["relation_list"]:
                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                rel["subj_tok_span"] = char_span2tok_span(subj_char_span, char2tok_span)
                rel["obj_tok_span"] = char_span2tok_span(obj_char_span, char2tok_span)
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                ent["tok_span"] = char_span2tok_span(char_span, char2tok_span)
            if "event_list" in sample:
                for event in sample["event_list"]:
                    event["trigger_tok_span"] = char_span2tok_span(event["trigger_char_span"], char2tok_span)
                    for arg in event["argument_list"]:
                        arg["tok_span"] = char_span2tok_span(arg["char_span"], char2tok_span)
        return dataset


from tplinker import (HandshakingTaggingScheme, DataMaker4Bert, DataMaker4BiLSTM, TPLinkerBert, TPLinkerBiLSTM, MetricsCalculator)
# import wandb
import config
# from glove import Glove
import numpy as np

# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
# config = yaml.load(open("train_config.yaml", "r"), Loader = yaml.FullLoader)


config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

data_home = config["data_home"]
experiment_name = config["exp_name"]
train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])

if config["logger"] == "wandb":
    # init wandb
    wandb.init(project=experiment_name,
               name=config["run_name"],
               config=hyper_parameters  # Initialize config
               )

    wandb.config.note = config["note"]

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    logger = DefaultLogger(config["log_path"], experiment_name, config["run_name"], config["run_id"], hyper_parameters)
    model_state_dict_dir = config["path_to_save_model"]
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

# # Load Data
train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
"""
standard data format
{'text': "It will be the final movie credited to Debra Hill , a film producer and native of Haddonfield , who produced '' Halloween '' and was considered a pioneering woman in film .", 
'id': 'train_2', 
'relation_list': [{'subject': 'Debra Hill', 'object': 'Haddonfield', 'subj_char_span': [39, 49], 'obj_char_span': [82, 93], 'predicate': '/people/person/place_of_birth', 'subj_tok_span': [9, 11], 'obj_tok_span': [21, 22]}], 
'entity_list': [{'text': 'Debra Hill', 'type': 'DEFAULT', 'char_span': [39, 49], 'tok_span': [9, 11]}, {'text': 'Haddonfield', 'type': 'DEFAULT', 'char_span': [82, 93], 'tok_span': [21, 22]}]}
=============================================================================================================================
{'text': 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .', 
'id': 'train_0', 
'relation_list': [{'subject': 'Annandale-on-Hudson', 'object': 'Bard College', 'subj_char_span': [68, 87], 'obj_char_span': [53, 65], 'predicate': '/location/location/contains', 'subj_tok_span': [11, 16], 'obj_tok_span': [8, 10]}], 
'entity_list': [{'text': 'Annandale-on-Hudson', 'type': 'DEFAULT', 'char_span': [68, 87], 'tok_span': [11, 16]}, {'text': 'Bard College', 'type': 'DEFAULT', 'char_span': [53, 65], 'tok_span': [8, 10]}]}

{'text': 'In Baghdad , Mr. Gates talked to enlisted service members on the second day of his visit to Iraq .', 
'id': 'train_21', 
'relation_list': [{'subject': 'Iraq', 'object': 'Baghdad', 'subj_char_span': [92, 96], 'obj_char_span': [3, 10], 'predicate': '/location/country/capital', 'subj_tok_span': [24, 25], 'obj_tok_span': [1, 2]}, {'subject': 'Iraq', 'object': 'Baghdad', 'subj_char_span': [92, 96], 'obj_char_span': [3, 10], 'predicate': '/location/location/contains', 'subj_tok_span': [24, 25], 'obj_tok_span': [1, 2]}],
'entity_list': [{'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [92, 96], 'tok_span': [24, 25]}, {'text': 'Baghdad', 'type': 'DEFAULT', 'char_span': [3, 10], 'tok_span': [1, 2]}, {'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [92, 96], 'tok_span': [24, 25]}, {'text': 'Baghdad', 'type': 'DEFAULT', 'char_span': [3, 10], 'tok_span': [1, 2]}]}

{'text': 'The meeting , in the house of Sheik Hamid Turki al-Shawka , a prominent tribal leader from Ramadi , lasted for five hours and included Sunni Arabs from Qaim , near the Syrian border , Mosul , in northern Iraq , and Baquba , north of Baghdad , as well as some Kurds and a few Shiites , the leaders said .', 
'id': 'train_30', 
'relation_list': [{'subject': 'Iraq', 'object': 'Ramadi', 'subj_char_span': [204, 208], 'obj_char_span': [91, 97], 'predicate': '/location/location/contains', 'subj_tok_span': [55, 56], 'obj_tok_span': [26, 27]}, {'subject': 'Iraq', 'object': 'Baghdad', 'subj_char_span': [204, 208], 'obj_char_span': [233, 240], 'predicate': '/location/country/capital', 'subj_tok_span': [55, 56], 'obj_tok_span': [62, 63]}, {'subject': 'Iraq', 'object': 'Baghdad', 'subj_char_span': [204, 208], 'obj_char_span': [233, 240], 'predicate': '/location/location/contains', 'subj_tok_span': [55, 56], 'obj_tok_span': [62, 63]}, {'subject': 'Iraq', 'object': 'Mosul', 'subj_char_span': [204, 208], 'obj_char_span': [184, 189], 'predicate': '/location/location/contains', 'subj_tok_span': [55, 56], 'obj_tok_span': [50, 51]}], 
'entity_list': [{'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [204, 208], 'tok_span': [55, 56]}, {'text': 'Ramadi', 'type': 'DEFAULT', 'char_span': [91, 97], 'tok_span': [26, 27]}, {'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [204, 208], 'tok_span': [55, 56]}, {'text': 'Baghdad', 'type': 'DEFAULT', 'char_span': [233, 240], 'tok_span': [62, 63]}, {'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [204, 208], 'tok_span': [55, 56]}, {'text': 'Baghdad', 'type': 'DEFAULT', 'char_span': [233, 240], 'tok_span': [62, 63]}, {'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [204, 208], 'tok_span': [55, 56]}, {'text': 'Mosul', 'type': 'DEFAULT', 'char_span': [184, 189], 'tok_span': [50, 51]}]}

{'text': "Somewhat chastened by his retreat in the polls , Mr. Blair acknowledged that Britons had turned against him in part over accusations that he led them into a war in Iraq on dubious legal grounds and on the false premise that Saddam Hussein presented a direct threat because of a supposed arsenal of unconventional weapons that was never found . ''", 
'id': 'train_40', 
'relation_list': [{'subject': 'Saddam Hussein', 'object': 'Iraq', 'subj_char_span': [224, 238], 'obj_char_span': [164, 168], 'predicate': '/people/deceased_person/place_of_death', 'subj_tok_span': [70, 72], 'obj_tok_span': [50, 51]}, {'subject': 'Saddam Hussein', 'object': 'Iraq', 'subj_char_span': [224, 238], 'obj_char_span': [164, 168], 'predicate': '/people/person/place_of_birth', 'subj_tok_span': [70, 72], 'obj_tok_span': [50, 51]}, {'subject': 'Saddam Hussein', 'object': 'Iraq', 'subj_char_span': [224, 238], 'obj_char_span': [164, 168], 'predicate': '/people/person/nationality', 'subj_tok_span': [70, 72], 'obj_tok_span': [50, 51]}], 
'entity_list': [{'text': 'Saddam Hussein', 'type': 'DEFAULT', 'char_span': [224, 238], 'tok_span': [70, 72]}, {'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [164, 168], 'tok_span': [50, 51]}, {'text': 'Saddam Hussein', 'type': 'DEFAULT', 'char_span': [224, 238], 'tok_span': [70, 72]}, {'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [164, 168], 'tok_span': [50, 51]}, {'text': 'Saddam Hussein', 'type': 'DEFAULT', 'char_span': [224, 238], 'tok_span': [70, 72]}, {'text': 'Iraq', 'type': 'DEFAULT', 'char_span': [164, 168], 'tok_span': [50, 51]}]}
"""
valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))

# # Split
# @specific
if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False, do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
elif config["encoder"] in {"BiLSTM", }:
    tokenize = lambda text: text.split(" ")


    def get_tok2char_span_map(text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span

preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)

# train and valid max token num
max_tok_num = 0
all_data = train_data + valid_data

for sample in all_data:
    tokens = tokenize(sample["text"])
    max_tok_num = max(max_tok_num, len(tokens))
# max_tok_num


if max_tok_num > hyper_parameters["max_seq_len"]:
    train_data = preprocessor.split_into_short_samples(train_data,  # 56196    56957
                                                       hyper_parameters["max_seq_len"],
                                                       sliding_len=hyper_parameters["sliding_len"],
                                                       encoder=config["encoder"]
                                                       )
    valid_data = preprocessor.split_into_short_samples(valid_data,  # 5000   5069
                                                       hyper_parameters["max_seq_len"],
                                                       sliding_len=hyper_parameters["sliding_len"],
                                                       encoder=config["encoder"]
                                                       )

print("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))

# # Tagger (Decoder)


max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])
rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)  # important core Class of the structure to save future data

# # Dataset


if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False, do_lower_case=False)
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)  # important core Class：基于handshaking_tagger数据结构； DataMaker4Bert 生成的数据都将存储在基于handshaking_tagger的数据结构中

elif config["encoder"] in {"BiLSTM", }:
    token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
    token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
    idx2token = {idx: tok for tok, idx in token2idx.items()}


    def text2indices(text, max_seq_len):
        input_ids = []
        tokens = text.split(" ")
        for tok in tokens:
            if tok not in token2idx:
                input_ids.append(token2idx['<UNK>'])
            else:
                input_ids.append(token2idx[tok])
        if len(input_ids) < max_seq_len:
            input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
        input_ids = torch.tensor(input_ids[:max_seq_len])
        return input_ids


    data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)
indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)

train_dataloader = DataLoader(MyDataset(indexed_train_data),
                              batch_size=hyper_parameters["batch_size"],
                              shuffle=True,
                              num_workers=6,
                              drop_last=False,
                              collate_fn=data_maker.generate_batch,  # todo Train data set trans to dataLoader to return batch data to do train；完成Train data向矩阵结构的填充
                              )
valid_dataloader = DataLoader(MyDataset(indexed_valid_data),
                              batch_size=hyper_parameters["batch_size"],
                              shuffle=True,
                              num_workers=6,
                              drop_last=False,
                              collate_fn=data_maker.generate_batch,
                              )

# # have a look at dataloader
# train_data_iter = iter(train_dataloader)
# batch_data = next(train_data_iter)
# text_id_list, text_list, batch_input_ids, \
# batch_attention_mask, batch_token_type_ids, \
# offset_map_list, batch_ent_shaking_tag, \
# batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_data

# print(text_list[0])
# print()
# print(tokenizer.decode(batch_input_ids[0].tolist()))
# print(batch_input_ids.size())
# print(batch_attention_mask.size())
# print(batch_token_type_ids.size())
# print(len(offset_map_list))
# print(batch_ent_shaking_tag.size())
# print(batch_head_rel_shaking_tag.size())
# print(batch_tail_rel_shaking_tag.size())


# # Model
if config["encoder"] == "BERT":
    encoder = AutoModel.from_pretrained(config["bert_path"])
    hidden_size = encoder.config.hidden_size
    fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hidden_size]).to(device)
    rel_extractor = TPLinkerBert(encoder,  # bert
                                 len(rel2id),
                                 hyper_parameters["shaking_type"],
                                 hyper_parameters["inner_enc_type"],
                                 hyper_parameters["dist_emb_size"],
                                 hyper_parameters["ent_add_dist"],
                                 hyper_parameters["rel_add_dist"],
                                 )

# elif config["encoder"] in {"BiLSTM", }:
#     glove = Glove()
#     glove = glove.load(config["pretrained_word_embedding_path"])
#
#     # prepare embedding matrix
#     word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
#     count_in = 0
#
#     # 在预训练词向量中的用该预训练向量
#     # 不在预训练集里的用随机向量
#     for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
#         if tok in glove.dictionary:
#             count_in += 1
#             word_embedding_init_matrix[ind] = glove.word_vectors[glove.dictionary[tok]]
#
#     print("{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(idx2token)))  # 命中预训练词向量的比例
#     word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)
#
#     fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hyper_parameters["dec_hidden_size"]]).to(device)
#     rel_extractor = TPLinkerBiLSTM(word_embedding_init_matrix,
#                                    hyper_parameters["emb_dropout"],
#                                    hyper_parameters["enc_hidden_size"],
#                                    hyper_parameters["dec_hidden_size"],
#                                    hyper_parameters["rnn_dropout"],
#                                    len(rel2id),
#                                    hyper_parameters["shaking_type"],
#                                    hyper_parameters["inner_enc_type"],
#                                    hyper_parameters["dist_emb_size"],
#                                    hyper_parameters["ent_add_dist"],
#                                    hyper_parameters["rel_add_dist"],
#                                    )

rel_extractor = rel_extractor.to(device)


# all_paras = sum(x.numel() for x in rel_extractor.parameters())
# enc_paras = sum(x.numel() for x in encoder.parameters())


# print(all_paras, enc_paras)
# print(all_paras - enc_paras)


# # Metrics


def bias_loss(weights=None):
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device)
    cross_en = nn.CrossEntropyLoss(weight=weights)
    return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))


loss_func = bias_loss()

metrics = MetricsCalculator(handshaking_tagger)


# # Train


# train step
def train_step(batch_train_data, optimizer, loss_weights):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data

        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                                                                                                                                      batch_attention_mask.to(device),
                                                                                                                                                      batch_token_type_ids.to(device),
                                                                                                                                                      batch_ent_shaking_tag.to(device),
                                                                                                                                                      batch_head_rel_shaking_tag.to(device),
                                                                                                                                                      batch_tail_rel_shaking_tag.to(device)
                                                                                                                                                      )

    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data

        batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                                                                                          batch_ent_shaking_tag.to(device),
                                                                                                          batch_head_rel_shaking_tag.to(device),
                                                                                                          batch_tail_rel_shaking_tag.to(device)
                                                                                                          )

    # zero the parameter gradients
    optimizer.zero_grad()

    if config["encoder"] == "BERT":
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids,
                                                                                                batch_attention_mask,
                                                                                                batch_token_type_ids,
                                                                                                )
    elif config["encoder"] in {"BiLSTM", }:
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids)

    w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
    loss = w_ent * loss_func(ent_shaking_outputs, batch_ent_shaking_tag) + w_rel * loss_func(head_rel_shaking_outputs, batch_head_rel_shaking_tag) + w_rel * loss_func(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

    loss.backward()
    optimizer.step()

    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                 batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs,
                                                      batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs,
                                                      batch_tail_rel_shaking_tag)

    return loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item()


# valid step
def valid_step(batch_valid_data):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                                                                                                                                      batch_attention_mask.to(device),
                                                                                                                                                      batch_token_type_ids.to(device),
                                                                                                                                                      batch_ent_shaking_tag.to(device),
                                                                                                                                                      batch_head_rel_shaking_tag.to(device),
                                                                                                                                                      batch_tail_rel_shaking_tag.to(device)
                                                                                                                                                      )

    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

        batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                                                                                          batch_ent_shaking_tag.to(device),
                                                                                                          batch_head_rel_shaking_tag.to(device),
                                                                                                          batch_tail_rel_shaking_tag.to(device)
                                                                                                          )

    with torch.no_grad():
        if config["encoder"] == "BERT":
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids,
                                                                                                    batch_attention_mask,
                                                                                                    batch_token_type_ids,
                                                                                                    )
        elif config["encoder"] in {"BiLSTM", }:
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids)

    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                 batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs,
                                                      batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs,
                                                      batch_tail_rel_shaking_tag)

    rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list,
                                  ent_shaking_outputs,
                                  head_rel_shaking_outputs,
                                  tail_rel_shaking_outputs,
                                  hyper_parameters["match_pattern"]
                                  )

    return ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item(), rel_cpg


max_f1 = 0.


def train_n_valid(train_dataloader, dev_dataloader, optimizer, scheduler, num_epoch):
    def train(dataloader, ep):
        # train
        rel_extractor.train()

        t_ep = time.time()
        start_lr = optimizer.param_groups[0]['lr']
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
            current_step = steps_per_ep * ep + batch_ind
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((len(rel2id) / z) * current_step / total_steps, (len(rel2id) / z))
            loss_weights = {"ent": w_ent, "rel": w_rel}

            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = train_step(batch_train_data, optimizer, loss_weights)
            scheduler.step()

            total_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            avg_loss = total_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind + 1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind + 1)

            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," + "lr: {}, batch_time: {}, total_time: {} -------------"

            print(batch_print_format.format(experiment_name, config["run_name"],
                                            ep + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            avg_loss,
                                            avg_ent_sample_acc,
                                            avg_head_rel_sample_acc,
                                            avg_tail_rel_sample_acc,
                                            optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                            ), end="")

            if config["logger"] == "wandb" and batch_ind % hyper_parameters["log_interval"] == 0:
                logger.log({
                    "train_loss": avg_loss,
                    "train_ent_seq_acc": avg_ent_sample_acc,
                    "train_head_rel_acc": avg_head_rel_sample_acc,
                    "train_tail_rel_acc": avg_tail_rel_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })

        if config["logger"] != "wandb":  # only log once for training if logger is not wandb
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

    def valid(dataloader, ep):
        # valid
        rel_extractor.eval()

        t_ep = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="Validating")):
            ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc, rel_cpg = valid_step(batch_valid_data)

            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            total_rel_correct_num += rel_cpg[0]
            total_rel_pred_num += rel_cpg[1]
            total_rel_gold_num += rel_cpg[2]

        avg_ent_sample_acc = total_ent_sample_acc / len(dataloader)
        avg_head_rel_sample_acc = total_head_rel_sample_acc / len(dataloader)
        avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(dataloader)

        rel_prf = metrics.get_prf_scores(total_rel_correct_num, total_rel_pred_num, total_rel_gold_num)

        log_dict = {
            "val_ent_seq_acc": avg_ent_sample_acc,
            "val_head_rel_acc": avg_head_rel_sample_acc,
            "val_tail_rel_acc": avg_tail_rel_sample_acc,
            "val_prec": rel_prf[0],
            "val_recall": rel_prf[1],
            "val_f1": rel_prf[2],
            "time": time.time() - t_ep,
        }
        logger.log(log_dict)
        pprint(log_dict)

        return rel_prf[2]

    for ep in range(num_epoch):
        train(train_dataloader, ep)
        valid_f1 = valid(valid_dataloader, ep)

        global max_f1
        if valid_f1 >= max_f1:
            max_f1 = valid_f1
            if valid_f1 > config["f1_2_save"]:  # save the best model
                modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                torch.save(rel_extractor.state_dict(), os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(modle_state_num)))
        #                 scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
        #                 torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir, "scheduler_state_dict_{}.pt".format(scheduler_state_num)))
        print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))


# optimizer
init_learning_rate = float(hyper_parameters["lr"])
optimizer = torch.optim.Adam(rel_extractor.parameters(), lr=init_learning_rate)

if hyper_parameters["scheduler"] == "CAWR":
    T_mult = hyper_parameters["T_mult"]
    rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)

elif hyper_parameters["scheduler"] == "Step":
    decay_rate = hyper_parameters["decay_rate"]
    decay_steps = hyper_parameters["decay_steps"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    rel_extractor.load_state_dict(torch.load(model_state_path))
    print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))

# train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, hyper_parameters["epochs"])
if __name__ == '__main__':
    train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, hyper_parameters["epochs"])
