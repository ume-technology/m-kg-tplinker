import string
import random

common = {
    # "exp_name": "nyt_star",
    "exp_name": "baidu_relation",  # todo change

    "device_num": 0,

    "rel2id": "rel2id.json",
    "encoder": "BERT",
    # "encoder": "BiLSTM",

    "hyper_parameters": {
        "shaking_type": "cat",  # cat, cat_plus, cln, cln_plus; Experiments show that cat/cat_plus work better with BiLSTM, while cln/cln_plus work better with BERT. The results in the paper are produced by "cat". So, if you want to reproduce the results, "cat" is enough, no matter for BERT or BiLSTM.
        "inner_enc_type": "lstm",  # valid only if cat_plus or cln_plus is set. It is the way how to encode inner tokens between each token pairs. If you only want to reproduce the results, just leave it alone.
        "dist_emb_size": -1,  # -1: do not use distance embedding; other number: need to be larger than the max_seq_len of the inputs. set -1 if you only want to reproduce the results in the paper.
        "ent_add_dist": False,  # set true if you want add distance embeddings for each token pairs. (for entity decoder)
        "rel_add_dist": False,  # the same as above (for relation decoder)

        # "match_pattern": "only_head_text",  # only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span
        "match_pattern": "only_head_index",  # only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span  # todo change  DuIE use only_head_index
    }
}

common["run_name"] = "{}+{}+{}".format("TP1", common["hyper_parameters"]["shaking_type"], common["encoder"]) + ""

run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
train_config = {
    "train_data": "/mnt/f/Pictures/m-kg-tplinker/m_kg_tplinker_original/baidu_relation/duie-data/in_data_data4bert/train_data.json",  # todo change
    # "train_data": r"F:\Pictures\m-kg-tplinker\m_kg_tplinker_original\baidu_relation\duie-data\in_data_data4bert\train_data.json",  # todo change
    "valid_data": "/mnt/f/Pictures/m-kg-tplinker/m_kg_tplinker_original/baidu_relation/duie-data/in_data_data4bert/valid_data.json",
    # "valid_data": r"F:\Pictures\m-kg-tplinker\m_kg_tplinker_original\baidu_relation\duie-data\in_data_data4bert\valid_data.json",

    "rel2id": "rel2id.json",
    # "logger": "wandb",  # if wandb, comment the following four lines
    # if logger is set as default, uncomment the following four lines
    "logger": "default",
    "run_id": run_id,
    "log_path": "./default_log_dir/default.log",
    "path_to_save_model": "./default_log_dir/{}".format(run_id),

    "f1_2_save": 0,  # only save the model state dict if F1 score surpasses <f1_2_save>
    "fr_scratch": True,  # whether train_config from scratch
    "note": "start from scratch",  # write down notes here if you want, it will be logged
    "model_state_dict_path": "",  # if not fr scratch, set a model_state_dict
    "hyper_parameters": {
        "batch_size": 6,  # 6 for webnlg and webnlg_star
        "epochs": 100,
        "seed": 2333,
        "log_interval": 10,
        "max_seq_len": 300,
        "sliding_len": 20,
        # to speed up the training process, the loss of EH-to-ET sequence is set higher than other sequences at the beginning, but it will recover in <loss_weight_recover_steps> steps.
        "loss_weight_recover_steps": 6000,
        "scheduler": "CAWR",  # Step
    },
}

eval_config = {
    "model_state_dict_dir": "./wandb",  # if use wandb, set "./wandb", or set "./default_log_dir" if you use default logger
    "run_ids": ["10suiyrf", ],
    "last_k_model": 1,
    "test_data": "*test*.json",  # "*test*.json"

    # where to save results
    "save_res": False,
    "save_res_dir": "../results",

    # score: set true only if test set is annotated with ground truth
    "score": True,

    "hyper_parameters": {
        "batch_size": 32,
        "force_split": False,
        "max_test_seq_len": 512,
        "sliding_len": 50,
    },
}

bert_config = {
    "data_home": "/mnt/f/Pictures/m-kg-tplinker/m_kg_tplinker_original/baidu_relation/duie-data/in_data_data4bert",
    # "data_home": r"F:\Pictures\m-kg-tplinker\m_kg_tplinker_original\baidu_relation\duie-data\in_data_data4bert",
    "bert_path": "/mnt/f/Pictures/bigfiles/premodelfiles/hflchinese-bert-wwm-ext",  # todo  change
    # "bert_path": r"F:\Pictures\bigfiles\premodelfiles\hflchinese-bert-wwm-ext",
    "hyper_parameters": {"lr": 5e-5, },
}
bilstm_config = {
    "data_home": "../data4bilstm",
    "token2idx": "token2idx.json",
    "pretrained_word_embedding_path": "../../pretrained_emb/glove_300_nyt.emb",
    "hyper_parameters": {
        "lr": 1e-3,
        "enc_hidden_size": 300,
        "dec_hidden_size": 600,
        "emb_dropout": 0.1,
        "rnn_dropout": 0.1,
        "word_embedding_dim": 300,
    },
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------dicts above is all you need to set---------------------------------------------------
if common["encoder"] == "BERT":
    hyper_params = {**common["hyper_parameters"], **bert_config["hyper_parameters"]}
    common = {**common, **bert_config}
    common["hyper_parameters"] = hyper_params
elif common["encoder"] == "BiLSTM":
    hyper_params = {**common["hyper_parameters"], **bilstm_config["hyper_parameters"]}
    common = {**common, **bilstm_config}
    common["hyper_parameters"] = hyper_params

hyper_params = {**common["hyper_parameters"], **train_config["hyper_parameters"]}
train_config = {**train_config, **common}
train_config["hyper_parameters"] = hyper_params
if train_config["hyper_parameters"]["scheduler"] == "CAWR":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **cawr_scheduler}
elif train_config["hyper_parameters"]["scheduler"] == "Step":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **step_scheduler}

hyper_params = {**common["hyper_parameters"], **eval_config["hyper_parameters"]}
eval_config = {**eval_config, **common}
eval_config["hyper_parameters"] = hyper_params
