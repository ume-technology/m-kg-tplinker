# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:UmeAI
@File:BuildData.py
@Time:2023/1/10 15:06
@Read:
"""
import json
import os
from tqdm import tqdm
import re
from transformers import BertTokenizerFast
import copy
import torch
from m_kg_tplinker_original.TPlinker_joint_extraction.common.utils import Preprocessor
import yaml
import logging
from pprint import pprint

# from IPython.core.debugger import set_trace
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
config = yaml.load(open("build_data_config.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)
exp_name = config["exp_name"]
data_in_dir = os.path.join(config["data_in_dir"], exp_name)
data_out_dir = os.path.join(config["data_out_dir"], exp_name)
if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)

# Load Data
file_name2data = {}
for path, folds, files in os.walk(data_in_dir):
    for file_name in files:
        file_path = os.path.join(path, file_name)
        file_name = re.match("(.*?)\.json", file_name).group(1)
        # old function - wrong function
        # file_name2data[file_name] = json.load(open(file_path, "r", encoding="utf-8"))
        # todo read line from files
        _ = []
        for line in open(file_path, "r", encoding="utf-8"):
            line = json.loads(line)
            _.append(line)
        file_name2data[file_name] = _

# Preprocess
# @specific
tokenize = ''
preprocessor = ''
if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False, do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)
elif config["encoder"] == "BiLSTM":
    tokenize = lambda text: text.split(" ")


    def get_tok2char_span_map(text):
        tokens = tokenize(text)
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span


    preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)

# Transform  todo 得到实体和实体（实体有label）之间的关系
"""
train-data:
{'sentText': 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .',
 'articleId': '/m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006.backup/1669365.xml.pb', 
 'relationMentions': [{'em1Text': 'Annandale-on-Hudson', 'em2Text': 'Bard College', 'label': '/location/location/contains'}], 
 'entityMentions': [{'start': 1, 'label': 'ORGANIZATION', 'text': 'Bard College'}, {'start': 2, 'label': 'LOCATION',  'text': 'Annandale-on-Hudson'}], 
 'sentId': '1'}
Transform to
{'text': 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .', 
'id': 'train_0',
'relation_list': [{'subject': 'Annandale-on-Hudson', 'predicate': '/location/location/contains', 'object': 'Bard College'}]}

valid-data:
{'sentText': 'In Queens , North Shore Towers , near the Nassau border , supplanted a golf course , and housing replaced a gravel quarry in Douglaston .', 
'articleId': '/m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006.backup/1719412.xml.pb', 
'relationMentions': [{'em1Text': 'Douglaston', 'em2Text': 'Queens', 'label': '/location/neighborhood/neighborhood_of'}, {'em1Text': 'Queens', 'em2Text': 'Douglaston', 'label': '/location/location/contains'}], 
'entityMentions': [{'start': 0, 'label': 'LOCATION', 'text': 'Queens'}, {'start': 1, 'label': 'LOCATION', 'text': 'Nassau'}, {'start': 2, 'label': 'LOCATION', 'text': 'Douglaston'}], 
'sentId': '2'}
Transform to
{'text': 'In Queens , North Shore Towers , near the Nassau border , supplanted a golf course , and housing replaced a gravel quarry in Douglaston .', 
'id': 'valid_0', 
'relation_list': [{'subject': 'Douglaston', 'predicate': '/location/neighborhood/neighborhood_of', 'object': 'Queens'}, {'subject': 'Queens', 'predicate': '/location/location/contains', 'object': 'Douglaston'}]}

test-data:
{'sentText': 'But that spasm of irritation by a master intimidator was minor compared with what Bobby Fischer , the erratic former world chess champion , dished out in March at a news conference in Reykjavik , Iceland .', 
'articleId': '/m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006.backup/1677367.xml.pb', 
'relationMentions': [{'em1Text': 'Bobby Fischer', 'em2Text': 'Iceland', 'label': '/people/person/nationality'}, {'em1Text': 'Iceland', 'em2Text': 'Reykjavik', 'label': '/location/country/capital'}, {'em1Text': 'Iceland', 'em2Text': 'Reykjavik', 'label': '/location/location/contains'}, {'em1Text': 'Bobby Fischer', 'em2Text': 'Reykjavik', 'label': '/people/deceased_person/place_of_death'}], 
'entityMentions': [{'start': 0, 'label': 'PERSON', 'text': 'Bobby Fischer'}, {'start': 1, 'label': 'LOCATION', 'text': 'Reykjavik'}, {'start': 2, 'label': 'LOCATION', 'text': 'Iceland'}], 
'sentId': '1'}
Transform to
{'text': 'But that spasm of irritation by a master intimidator was minor compared with what Bobby Fischer , the erratic former world chess champion , dished out in March at a news conference in Reykjavik , Iceland .', 
'id': 'test_0', 
'relation_list': [{'subject': 'Bobby Fischer', 'predicate': '/people/person/nationality', 'object': 'Iceland'}, {'subject': 'Iceland', 'predicate': '/location/country/capital', 'object': 'Reykjavik'}, {'subject': 'Iceland', 'predicate': '/location/location/contains', 'object': 'Reykjavik'}, {'subject': 'Bobby Fischer', 'predicate': '/people/deceased_person/place_of_death', 'object': 'Reykjavik'}]}
"""
ori_format = config["ori_data_format"]
if ori_format != "tplinker":  # if tplinker, skip transforming
    for file_name, data in file_name2data.items():
        data_type = ''
        if "train" in file_name:
            data_type = "train"
        if "valid" in file_name:
            data_type = "valid"
        if "test" in file_name:
            data_type = "test"
        data = preprocessor.transform_data(data, ori_format=ori_format, dataset_type=data_type, add_id=True)
        file_name2data[file_name] = data


#  Clean and Add Spans
# check token level span
def check_tok_span(data):
    def extr_ent(text, tok_span, tok2char_span):
        char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
        char_span = (char_span_list[0][0], char_span_list[-1][1])
        decoded_ent = text[char_span[0]:char_span[1]]
        return decoded_ent

    span_error_memory = set()
    for sample in tqdm(data, desc="check tok spans"):
        text = sample["text"]
        tok2char_span = get_tok2char_span_map(text)
        for ent in sample["entity_list"]:
            tok_span = ent["tok_span"]
            if extr_ent(text, tok_span, tok2char_span) != ent["text"]:
                span_error_memory.add("extr ent: {}---gold ent: {}".format(extr_ent(text, tok_span, tok2char_span), ent["text"]))

        for rel in sample["relation_list"]:
            subj_tok_span, obj_tok_span = rel["subj_tok_span"], rel["obj_tok_span"]
            if extr_ent(text, subj_tok_span, tok2char_span) != rel["subject"]:
                span_error_memory.add("extr: {}---gold: {}".format(extr_ent(text, subj_tok_span, tok2char_span), rel["subject"]))
            if extr_ent(text, obj_tok_span, tok2char_span) != rel["object"]:
                span_error_memory.add("extr: {}---gold: {}".format(extr_ent(text, obj_tok_span, tok2char_span), rel["object"]))

    return span_error_memory


# clean, add char span, tok span
# collect relations
# check tok spans
# todo save entity label relations
rel_set = set()
ent_set = set()
error_statistics = {}
for file_name, data in file_name2data.items():
    assert len(data) > 0
    if "relation_list" in data[0]:  # train or valid data  # todo important sequence 中包含的类型下的实体是否存在关系
        # rm redundant whitespaces
        # separate by whitespaces
        data = preprocessor.clean_data_wo_span(data, separate=config["separate_char_by_white"])
        error_statistics[file_name] = {}
        #         if file_name != "train_data":
        #             set_trace()
        # add char span
        if config["add_char_span"]:
            data, miss_sample_list = preprocessor.add_char_span(data, config["ignore_subword"])
            error_statistics[file_name]["miss_samples"] = len(miss_sample_list)

        #         # clean
        #         data, bad_samples_w_char_span_error = preprocessor.clean_data_w_span(data)
        #         error_statistics[file_name]["char_span_error"] = len(bad_samples_w_char_span_error)

        # collect relation types and entity types
        for sample in tqdm(data, desc="building relation type set and entity type set"):
            if "entity_list" not in sample:  # if "entity_list" not in sample, generate entity list with default type
                ent_list = []
                for rel in sample["relation_list"]:
                    ent_list.append({
                        "text": rel["subject"],
                        "type": "DEFAULT",
                        "char_span": rel["subj_char_span"],
                    })
                    ent_list.append({
                        "text": rel["object"],
                        "type": "DEFAULT",
                        "char_span": rel["obj_char_span"],
                    })
                sample["entity_list"] = ent_list

            for ent in sample["entity_list"]:
                ent_set.add(ent["type"])

            for rel in sample["relation_list"]:
                rel_set.add(rel["predicate"])

        # add tok span
        data = preprocessor.add_tok_span(data)

        # check tok span
        if config["check_tok_span"]:
            span_error_memory = check_tok_span(data)
            if len(span_error_memory) > 0:
                print(span_error_memory)
            error_statistics[file_name]["tok_span_error"] = len(span_error_memory)

        # todo final get and save token/char index information
        file_name2data[file_name] = data
pprint(error_statistics)

# Output to Disk
rel_set = sorted(rel_set)
rel2id = {rel: ind for ind, rel in enumerate(rel_set)}

ent_set = sorted(ent_set)
ent2id = {ent: ind for ind, ent in enumerate(ent_set)}

data_statistics = {
    "relation_type_num": len(rel2id),
    "entity_type_num": len(ent2id),
}

for file_name, data in file_name2data.items():
    data_path = os.path.join(data_out_dir, "{}.json".format(file_name))
    json.dump(data, open(data_path, "w", encoding="utf-8"), ensure_ascii=False)
    logging.info("{} is output to {}".format(file_name, data_path))
    data_statistics[file_name] = len(data)

rel2id_path = os.path.join(data_out_dir, "rel2id.json")
json.dump(rel2id, open(rel2id_path, "w", encoding="utf-8"), ensure_ascii=False)
logging.info("rel2id is output to {}".format(rel2id_path))

ent2id_path = os.path.join(data_out_dir, "ent2id.json")
json.dump(ent2id, open(ent2id_path, "w", encoding="utf-8"), ensure_ascii=False)
logging.info("ent2id is output to {}".format(ent2id_path))

data_statistics_path = os.path.join(data_out_dir, "data_statistics.txt")
json.dump(data_statistics, open(data_statistics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
logging.info("data_statistics is output to {}".format(data_statistics_path))

pprint(data_statistics)

# Genrate WordDict
if config["encoder"] in {"BiLSTM", }:
    all_data = []
    for data in list(file_name2data.values()):
        all_data.extend(data)

    token2num = {}
    for sample in tqdm(all_data, desc="Tokenizing"):
        text = sample['text']
        for tok in tokenize(text):
            token2num[tok] = token2num.get(tok, 0) + 1

    token2num = dict(sorted(token2num.items(), key=lambda x: x[1], reverse=True))
    max_token_num = 50000
    token_set = set()
    for tok, num in tqdm(token2num.items(), desc="Filter uncommon words"):
        if num < 3:  # filter words with a frequency of less than 3
            continue
        token_set.add(tok)
        if len(token_set) == max_token_num:
            break

    token2idx = {tok: idx + 2 for idx, tok in enumerate(sorted(token_set))}
    token2idx["<PAD>"] = 0
    token2idx["<UNK>"] = 1
    #     idx2token = {idx:tok for tok, idx in token2idx.items()}

    dict_path = os.path.join(data_out_dir, "token2idx.json")
    json.dump(token2idx, open(dict_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    logging.info("token2idx is output to {}, total token num: {}".format(dict_path, len(token2idx)))
