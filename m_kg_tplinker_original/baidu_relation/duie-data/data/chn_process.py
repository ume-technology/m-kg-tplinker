# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:UmeAI
@File:chn_process.py
@Time:2023/1/12 15:54
@Read: 基于DUIE的中文数据格式化处理
"""
format = {
    "id": 9,
    "text": "在导师阵容方面，英达有望联手《中国喜剧王》选拔新一代笑星",
    "relation_list": [
        {
            "subject": "中国喜剧王",
            "object": "英达",
            "subj_char_span": [
                15,
                20
            ],
            "obj_char_span": [
                8,
                10
            ],
            "predicate": "嘉宾",
            "subj_tok_span": [
                15,
                20
            ],
            "obj_tok_span": [
                8,
                10
            ]
        }
    ],
    "entity_list": [
        {
            "text": "英达",
            "type": "人物",
            "char_span": [
                8,
                10
            ],
            "tok_span": [
                8,
                10
            ]
        },
        {
            "text": "中国喜剧王",
            "type": "电视综艺",
            "char_span": [
                15,
                20
            ],
            "tok_span": [
                15,
                20
            ]
        }
    ]
}

import json

with open('mid_train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open('mid_valid_data.json', 'r', encoding='utf-8') as f:
    valid_data = json.load(f)

# todo 生成标准数据
# for idx, _ in enumerate([train_data, valid_data]):
#     if idx == 0:
#         continue
#     for i in _:
#         text = i['text'].replace(' ', '').strip().replace('　', '')
#         i['text'] = text
#         relation_list = i['relation_list']
#         entity_list = i['entity_list']
#         for ii in relation_list:
#             obj = ii['object'].strip().replace(' ', '').replace('　', '')
#             sub = ii['subject'].strip().replace(' ', '').replace('　', '')
#             ii['object'] = obj
#             ii['subject'] = sub
#             obj_h = text.index(obj)
#             obj_t = obj_h + len(obj)
#             sub_h = text.index(sub)
#             sub_t = sub_h + len(sub)
#             subj_char_span = [sub_h, sub_t]
#             subj_tok_span = [sub_h, sub_t]
#             obj_char_span = [obj_h, obj_t]
#             obj_tok_span = [obj_h, obj_t]
#             ii['subj_char_span'] = subj_char_span
#             ii['subj_tok_span'] = subj_tok_span
#             ii['obj_char_span'] = obj_char_span
#             ii['obj_tok_span'] = obj_tok_span
#             for iii in entity_list:
#                 if obj in iii.values():
#                     iii['char_span'] = subj_char_span
#                     iii['tok_span'] = subj_tok_span
#                 if sub in iii.values():
#                     iii['char_span'] = obj_char_span
#                     iii['tok_span'] = obj_tok_span
#     with open('../in_data_data4bert/valid_data.json', 'w', encoding='utf-8')as f:
#         json.dump(_, f, ensure_ascii=False)

# rels = set()
# ents = set()
# for i in train_data:
#     rel = i['relation_list']
#     ent = i['entity_list']
#     for ii in rel:
#         rels.add(ii['predicate'])
#     for ii in ent:
#         ents.add(ii['type'])
#
# entss = {}
# for idx, i in enumerate(ents):
#     entss[i] = idx
# with open('../in_data_data4bert/ent2id.json', 'w', encoding='utf-8') as f:
#     json.dump(entss, f, ensure_ascii=False)
#
# relss = {}
# for idx, i in enumerate(rels):
#     relss[i] = idx
# with open('../in_data_data4bert/rel2id.json', 'w', encoding='utf-8') as f:
#     json.dump(relss, f, ensure_ascii=False)
