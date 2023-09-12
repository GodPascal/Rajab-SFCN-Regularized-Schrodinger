import torch
import torch.nn as nn

import os
from os import makedirs, path

import random
import numpy as np

import socket
from datetime import datetime

from statistics import mean, stdev
import pickle


def parse_config(config, readonly=False):
    
    # for entry_level_1 in ["logging", "dataset", "dataloader", "model", "misc"]:
    #     if entry_level_1 not in config:
    #         print("ERROR: No {} is defined in \"config\"!".format(entry_level_1))    
    #         exit()

    
    # for dataset in config["datasets"]:
    #     if config["datasets"]["internal_train"]["type"] == "internal_train":
    #         for entry in ["", ""]:
    #             if entry not in config["datasets"][dataset]:
    #                 print("ERROR: No {} is defined in \"config[\"datasets\"][{}]\"!".format(entry, dataset))    
    #                 exit()

    if config["model"]["resume_training"] == True:
        try:    
            with open(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/config.pkl", "rb") as f:
                config = pickle.load(f)
            config["model"]["resume_training"] = True
            return config
        except:
            print("[ERR] No \"config\" file found to load!")
            exit()

    if config["logging"]["run_id"] == "" or config["logging"]["run_id"] == "auto":
        config["logging"]["run_id"] = socket.gethostname().split(".")[0] # Machine name
        config["logging"]["run_id"] += "-" + datetime.today().strftime("%y.%m.%d") # Today date
        #config["logging"]["run_id"] += "-ll" + "{}".format(config["model"]["hyperparameters"]["num_last_layer_features"])  # Last Layer size
        config["logging"]["run_id"] += "-bt" + "{}".format(config["model"]["hyperparameters"]["num_latent_factors"])  # Last Layer size
        config["logging"]["run_id"] += "-bs" + "{}".format(config["dataloader"]["batch_size"])  # Batch size
        config["logging"]["wandb"]["run_id"] = config["logging"]["run_id"]

    if config["logging"]["wandb"]["project_name"] == "":
        config["logging"]["wandb"]["project_name"] = config["logging"]["project_name"]

    if config["logging"]["wandb"]["run_id"] == "":
        config["logging"]["wandb"]["run_id"] = config["logging"]["run_id"]

    risk_factors_to_calculate_mean_std = {}
    for risk_factor in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
        if config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["type"] == "continuous" and \
            (config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["mean"] == -1 or config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["stdev"] == -1):
            risk_factors_to_calculate_mean_std[risk_factor] = []
    risk_factors_to_calculate_class_balance = {}
    for risk_factor in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
        if config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["type"] == "discrete" and \
            (len(config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["class_balance"]) == 0):
            risk_factors_to_calculate_class_balance[risk_factor] = []
    if len(risk_factors_to_calculate_mean_std) > 0 or len(risk_factors_to_calculate_class_balance) > 0:
        metadata = np.load(config["datasets"]["internal_train"]["metadata_filepath"], allow_pickle=True)
        metadata = metadata["data"]
        for sample in metadata:
            for risk_factor in risk_factors_to_calculate_mean_std:
                risk_factors_to_calculate_mean_std[risk_factor].append(int(sample[risk_factor]))
            for risk_factor in risk_factors_to_calculate_class_balance:
                risk_factors_to_calculate_class_balance[risk_factor].append(config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["labels"].index(sample[risk_factor]))
        for risk_factor in risk_factors_to_calculate_mean_std:
            config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["mean"] = mean(risk_factors_to_calculate_mean_std[risk_factor])
            config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["stdev"] = stdev(risk_factors_to_calculate_mean_std[risk_factor])
            q1 = np.percentile(risk_factors_to_calculate_mean_std[risk_factor], 25)
            q3 = np.percentile(risk_factors_to_calculate_mean_std[risk_factor], 75)
            iqr = q3 - q1
            config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["normalized_lower_outer_fence"] = ((q1 - 3*iqr) - config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["mean"]) / config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["stdev"]
            config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["normalized_upper_outer_fence"] = ((q3 + 3*iqr) - config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["mean"]) / config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["stdev"]
        for risk_factor in risk_factors_to_calculate_class_balance:
            no_of_labels = len(config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["labels"])
            config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["class_balance"] = [0.0] * no_of_labels
            for label_index in range(no_of_labels):
                config["model"]["hyperparameters"]["regularizers"]["risk_factors"][risk_factor]["class_balance"][label_index] = risk_factors_to_calculate_class_balance[risk_factor].count(label_index) / len(risk_factors_to_calculate_class_balance[risk_factor])

    if config["misc"]["device"] == "cuda" and not(torch.cuda.is_available()):
        print("[ERR] Device is set to CUDA and CUDA is not available!")
        exit()

    if readonly == False:
        if not path.exists(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/results/"):
            makedirs(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/results/")
        if not path.exists(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/errors/"):
            makedirs(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/errors/")
        if not path.exists(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/"):
            makedirs(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/")
        with open(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/config.pkl", "wb") as f:
            pickle.dump(config, f)

    return config


def init_torch(deterministic=False, optimize_performance=True, debugging=False):

    if debugging:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    if optimize_performance:
        torch.autograd.anomaly_mode.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)
        if not(deterministic):
            torch.backends.cudnn.benchmark = True

    return 0


def init_random_seed(for_python=42, for_numpy=42, for_torch=42):
    
    random.seed(for_python)
    np.random.seed(for_numpy)
    torch.manual_seed(for_torch)


def init_model_weights(model):

    for m in model.modules():

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
            
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
        if isinstance(m, nn.GroupNorm):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


def init_gradient_clipping(model, min=-0.1, max=0.1):

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, min, max))


