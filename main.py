import torch

from matplotlib.pyplot import imshow
from matplotlib import figure
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import seaborn as sns
sns.set_style("darkgrid")

from os import path

from model import RajabSFCNVAERegularizedLM2
from data import MRIDataset

import math
import numpy as np

from config import config
from config import on_computecanada

import init

from utils import create_canvas

from datetime import datetime

import gc

config = init.parse_config(config)
init.init_torch()
init.init_random_seed()

## Model Initialization
device = torch.device(config["misc"]["device"])
affine_transforms_list = ["scale_x", "scale_y", "scale_z",
                     "rotation_x", "rotation_y", "rotation_z",
                     "translation_x", "translation_y", "translation_z"]
affine_transforms_error_coeff = config["model"]["hyperparameters"]["regularizers"]["risk_factors"]["age"]["error_coeff"]
model = RajabSFCNVAERegularizedLM2(config["datasets"]["internal_train"]["input_shape"],
    config["dataloader"]["batch_size"],
    config["model"]["hyperparameters"]["dropout_last_layer"],
    risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"],
    affine_transforms=affine_transforms_list,
    variatioanl_bottleneck_size=config["model"]["hyperparameters"]["num_latent_factors"],
    num_rf_ensembles=config["model"]["hyperparameters"]["num_rf_ensembles"]).to(device)
if config["model"]["hyperparameters"]["gradient_clipping"]:
    init.init_gradient_clipping(model, config["model"]["hyperparameters"]["gradient_min"], config["model"]["hyperparameters"]["gradient_max"])
# init.init_model_weights(model)


if config["logging"]["wandb"]["enabled"]:
    import wandb

    wandbconfig = {
        "architecture": "sfcn",
        "weight_initialization": "kaiming, normal, relu",
        "optimizer": "SGD",
        "scheduler": "StepLR",
        "augmentation": "Louis",
        "config": config        
    }

    if config["model"]["resume_training"]:
        wandb.init(project=config["logging"]["wandb"]["project_name"], entity=config["logging"]["wandb"]["entity"], id=config["logging"]["wandb"]["run_id"], config=wandbconfig, resume="allow")
    else:
        wandb.init(project=config["logging"]["wandb"]["project_name"], entity=config["logging"]["wandb"]["entity"], id=config["logging"]["wandb"]["run_id"], config=wandbconfig)


train_data = MRIDataset(config["datasets"]["internal_train"]["path"],
    config["datasets"]["internal_train"]["metadata_filepath"],
    num_subjects=config["datasets"]["internal_train"]["num_subjects"],
    k=config["datasets"]["internal_train"]["num_train_folds"],
    fold=config["datasets"]["internal_train"]["train_folds"],
    risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"])

validation_data = []
for f in range(len(config["datasets"]["internal_validation"]["train_folds"])):
    validation_data.append(MRIDataset(config["datasets"]["internal_validation"]["path"],
        config["datasets"]["internal_validation"]["metadata_filepath"],
        num_subjects=config["datasets"]["internal_validation"]["num_subjects"],
        k=config["datasets"]["internal_validation"]["num_train_folds"],
        fold=[config["datasets"]["internal_validation"]["train_folds"][f]],
        risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"],
        test_mode=True))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["dataloader"]["batch_size"], num_workers=config["dataloader"]["num_workers"], shuffle=True, drop_last=True)

validation_loaders = []
for i in range(len(config["datasets"]["internal_validation"]["train_folds"])):
    validation_loaders.append(torch.utils.data.DataLoader(validation_data[i], batch_size=config["dataloader"]["batch_size"], num_workers=config["dataloader"]["num_workers"], shuffle=False, drop_last=True))

scaler = torch.cuda.amp.GradScaler()
optimizer = optim.SGD(model.parameters(), lr=config["model"]["hyperparameters"]["lr"], weight_decay=config["model"]["hyperparameters"]["l2_weight_decay"])
scheduler = StepLR(optimizer,
                   step_size=config["model"]["hyperparameters"]["lr_decay_step_size"],
                   gamma=config["model"]["hyperparameters"]["lr_decay_gamma"])

number_of_iterations_per_epoch = math.ceil(train_data.__len__() / config["dataloader"]["batch_size"])
current_epoch = 0
beta = 0

if config["model"]["resume_training"]:
    if path.exists(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/"  + "model.eLast") == False:
        print("[ERR] No model found to resume from!")
        exit()
    print("[INF] Loading the last saved model to resume training from...")
    last_state = torch.load(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/"  + "model.eLast")

    model.load_state_dict(last_state["model"])
    optimizer.load_state_dict(last_state["optimizer"])
    scheduler.load_state_dict(last_state["scheduler"])
    current_epoch = last_state["current_epoch"]
    del last_state
    print("[INF] Last model was loaded successfully!")


print(sum(p.numel() for p in model.parameters() if p.requires_grad))

if config["logging"]["wandb"]["enabled"]:
    wandb.watch(model, log_freq=1)    


for epoch in range(current_epoch, config["model"]["hyperparameters"]["num_epochs"]):

    start = datetime.now()
    
    init.init_random_seed(for_python=epoch, for_numpy=epoch, for_torch=epoch)
    train_loss = {}

    ## Training
    model.train()
    for idx, train_batch in enumerate(train_loader):

        #gc.collect()

        t1w = train_batch["t1w"].to(device)
        t1w_mask = train_batch["t1w_mask"].to(device)

        risk_factors = {}
        for rf in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
            risk_factors[rf] = train_batch[rf].to(device)
        affine_transforms = {}
        for transform in affine_transforms_list:
            affine_transforms[transform] = train_batch[transform].to(device)
        
        optimizer.zero_grad()
        # for param in model.parameters():
        #     param.grad = None

        with torch.cuda.amp.autocast():
            estimated_risk_factors, estimated_affine_transforms, recon, _, mean, logvar = model(t1w, affine_transforms)

            loss = {}

            criterion = torch.nn.MSELoss(reduction="sum")
            loss["recon"] = criterion(recon, t1w) / config["dataloader"]["batch_size"]
            loss["recon_normalized"] = (criterion(recon * t1w_mask, t1w * t1w_mask) / t1w_mask[t1w_mask != 0].sum()).detach()

            loss["kl"] = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
            tau = ((epoch * (number_of_iterations_per_epoch) + idx) % math.ceil(number_of_iterations_per_epoch * config["model"]["hyperparameters"]["num_epochs"] / config["model"]["hyperparameters"]["beta_num_cycles"])) / (number_of_iterations_per_epoch * config["model"]["hyperparameters"]["num_epochs"] / config["model"]["hyperparameters"]["beta_num_cycles"])
            if tau <= config["model"]["hyperparameters"]["beta_r"]:
                beta = config["model"]["hyperparameters"]["beta_max"] * tau * (1 / config["model"]["hyperparameters"]["beta_r"])
            else:
                beta = config["model"]["hyperparameters"]["beta_max"]
            if config["model"]["hyperparameters"]["beta_normalizing"]:
                beta *= config["model"]["hyperparameters"]["num_latent_factors"] / np.prod(config["dataloader"]["batch_size"])
            
            loss["total"] = config["model"]["hyperparameters"]["vae_error_coeff"] * (loss["recon"] + beta * loss["kl"])
            if math.isnan(loss["total"].item()):
                print("recon error")

            criterion = torch.nn.MSELoss(reduction="mean")
            for transform in affine_transforms:
                loss[transform] = criterion(estimated_affine_transforms[transform], affine_transforms[transform].reshape(-1, 1))
                loss[transform+"_mae"] = (torch.sum(abs(estimated_affine_transforms[transform] - affine_transforms[transform].reshape(-1, 1))) / estimated_affine_transforms[transform].shape[0])
                if "scale" in transform:
                    loss[transform+"_mae"] *= np.sqrt((1.02 - 0.98) ** 2 / 12)
                if "rotation" in transform:
                    loss[transform+"_mae"] *= np.sqrt((9 - -9) ** 2 / 12)
                if "translation" in transform:
                    loss[transform+"_mae"] *= np.sqrt((4 - -4) ** 2 / 12)

                loss["total"] += affine_transforms_error_coeff * loss[transform]
                if math.isnan(loss[transform].item()):
                    print(transform + " error")

            criterion = torch.nn.MSELoss(reduction="mean")
            for rf in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
                if config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["type"] == "continuous":
                    loss[rf] = []
                    loss[rf+"_mae"] = []
                    for e in range(config["model"]["hyperparameters"]["num_rf_ensembles"]):
                        if config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["skip_outliers"]:
                            indices = torch.logical_and(risk_factors[rf] > config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["normalized_lower_outer_fence"], risk_factors[rf] < config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["normalized_upper_outer_fence"])
                            risk_factors[rf] = risk_factors[rf][indices]
                            estimated_risk_factors[e][rf] = estimated_risk_factors[e][rf][indices, :]

                        loss_rf = criterion(estimated_risk_factors[e][rf], risk_factors[rf].reshape(-1, 1))
                        
                        loss["total"] += config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["error_coeff"] * loss_rf
                        if math.isnan(loss_rf.item()):
                            print(rf + " error")

                        loss[rf].append((criterion(estimated_risk_factors[e][rf], risk_factors[rf].reshape(-1, 1))).item())
                        loss[rf+"_mae"].append(((torch.sum(abs(estimated_risk_factors[e][rf] - risk_factors[rf].reshape(-1, 1))) / estimated_risk_factors[e][rf].shape[0]) * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["stdev"]).item())
                    
                    loss[rf] = np.mean(loss[rf])
                    loss[rf+"_mae"] = np.mean(loss[rf+"_mae"])
            
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            for rf in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
                if config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["type"] == "discrete":
                    loss[rf] = []
                    loss[rf+"_accuracy"] = []
                    for e in range(config["model"]["hyperparameters"]["num_rf_ensembles"]):
                        loss_rf = criterion(estimated_risk_factors[e][rf], risk_factors[rf])
                        
                        loss["total"] += config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["error_coeff"] * loss_rf
                        if math.isnan(loss_rf.item()):
                            print(rf + " error")

                        loss[rf].append((criterion(estimated_risk_factors[e][rf], risk_factors[rf])).item())
                        loss[rf+"_accuracy"].append((torch.sum(torch.argmax(estimated_risk_factors[e][rf], 1) == torch.argmax(risk_factors[rf], 1)) / config["dataloader"]["batch_size"]).item())
                    
                    loss[rf] = np.mean(loss[rf])
                    loss[rf+"_accuracy"] = np.mean(loss[rf+"_accuracy"])


        scaler.scale(loss["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        for l in loss:
            if l in train_loss:
                if type(loss[l]) is torch.Tensor:
                    train_loss[l] += loss[l].detach().to("cpu").item() / len(train_loader)
                else:
                    train_loss[l] += loss[l] / len(train_loader)
            else:
                if type(loss[l]) is torch.Tensor:
                    train_loss[l] = loss[l].detach().to("cpu").item() / len(train_loader)
                else:
                    train_loss[l] = loss[l] / len(train_loader)

        if not(on_computecanada):
            print(f"{idx}/{len(train_loader)}, {loss['total'].item():.7f}, {beta:.7f}", end="\r")
        
        if math.isnan(loss["total"].item()):
           print("[ERR] Possible gradient explosion for following ids: {}".format(train_batch["id"]))
           exit()

    end = datetime.now()
    print(end - start)

    ## Evaluation
    model.eval()
    with torch.no_grad():
        log = {}
        test_log = []
        for i_test, validation_loader in enumerate(validation_loaders):
            test_log.append({})

            mris_to_plot = []
            risk_factors_to_plot = []
            recons_to_plot = []
            est_risk_factors_to_plot = []
            fake_samples_to_plot = []
            
            for idx, validation_batch in enumerate(validation_loader):

                t1w = validation_batch["t1w"].to(device)
                t1w_mask = validation_batch["t1w_mask"].to(device)

                risk_factors = {}
                for rf in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
                    risk_factors[rf] = validation_batch[rf].to(device)
                affine_transforms = {}
                for transform in affine_transforms_list:
                    affine_transforms[transform] = validation_batch[transform].to(device)
                
                estimated_risk_factors, estimated_affine_transforms, recon, _, mean, logvar = model(t1w, affine_transforms)

                loss = {}

                criterion = torch.nn.MSELoss(reduction="sum")
                loss["recon"] = criterion(recon, t1w) / t1w.shape[0]
                loss["recon_normalized"] = (criterion(recon * t1w_mask, t1w * t1w_mask) / t1w_mask[t1w_mask != 0].sum()).detach()

                loss["kl"] = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))

                loss["total"] = config["model"]["hyperparameters"]["vae_error_coeff"] * (loss["recon"] + beta * loss["kl"])

                criterion = torch.nn.MSELoss(reduction="mean")
                for transform in affine_transforms:
                    loss[transform] = criterion(estimated_affine_transforms[transform], affine_transforms[transform].reshape(-1, 1))
                    loss[transform+"_mae"] = (torch.sum(abs(estimated_affine_transforms[transform] - affine_transforms[transform].reshape(-1, 1))) / estimated_affine_transforms[transform].shape[0])
                    if "scale" in transform:
                        loss[transform+"_mae"] *= np.sqrt((1.02 - 0.98) ** 2 / 12)
                    if "rotation" in transform:
                        loss[transform+"_mae"] *= np.sqrt((9 - -9) ** 2 / 12)
                    if "translation" in transform:
                        loss[transform+"_mae"] *= np.sqrt((4 - -4) ** 2 / 12)

                    loss["total"] += affine_transforms_error_coeff * loss[transform]

                criterion = torch.nn.MSELoss(reduction="mean")
                for rf in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
                    if config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["type"] == "continuous":
                        loss[rf] = []
                        loss[rf+"_mae"] = []
                        for e in range(config["model"]["hyperparameters"]["num_rf_ensembles"]):
                            loss_rf = criterion(estimated_risk_factors[e][rf], risk_factors[rf].reshape(-1, 1))
                            
                            loss["total"] += config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["error_coeff"] * loss_rf

                            loss[rf].append((criterion(estimated_risk_factors[e][rf], risk_factors[rf].reshape(-1, 1))).item())
                            loss[rf+"_mae"].append(((torch.sum(abs(estimated_risk_factors[e][rf] - risk_factors[rf].reshape(-1, 1))) / config["dataloader"]["batch_size"]) * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["stdev"]).item())
                        
                        loss[rf] = np.mean(loss[rf])
                        loss[rf+"_mae"] = np.mean(loss[rf+"_mae"])
                
                criterion = torch.nn.CrossEntropyLoss(reduction="mean")
                for rf in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
                    if config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["type"] == "discrete":
                        loss[rf] = []
                        loss[rf+"_accuracy"] = []
                        for e in range(config["model"]["hyperparameters"]["num_rf_ensembles"]):
                            loss_rf = criterion(estimated_risk_factors[e][rf], risk_factors[rf])
                            
                            loss["total"] += config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf]["error_coeff"] * loss_rf
                    
                            loss[rf].append((criterion(estimated_risk_factors[e][rf], risk_factors[rf])).item())
                            loss[rf+"_accuracy"].append((torch.sum(torch.argmax(estimated_risk_factors[e][rf], 1) == torch.argmax(risk_factors[rf], 1)) / config["dataloader"]["batch_size"]).item())
                        
                        loss[rf] = np.mean(loss[rf])
                        loss[rf+"_accuracy"] = np.mean(loss[rf+"_accuracy"])

                for l in loss:
                    if l in test_log[i_test]:
                        if type(loss[l]) is torch.Tensor:
                            test_log[i_test][l] += loss[l].detach().cpu().item() / len(validation_loader)
                        else:
                            test_log[i_test][l] += loss[l] / len(validation_loader)
                    else:
                        if type(loss[l]) is torch.Tensor:
                            test_log[i_test][l] = loss[l].detach().cpu().item() / len(validation_loader)
                        else:
                            test_log[i_test][l] = loss[l] / len(validation_loader)

                if idx * config["dataloader"]["batch_size"] < config["logging"]["num_reconstructed_samples_to_save"]:                    
                    mris_to_plot.append(validation_batch["t1w"].squeeze(1))

                    if len(risk_factors) > 0:
                        risk_factors_to_plot.append({})
                        for rf in risk_factors:
                            risk_factors_to_plot[-1][rf] = risk_factors[rf].reshape(config["dataloader"]["batch_size"], -1).detach().cpu()

                    recons_to_plot.append(recon.squeeze(1).detach().cpu())
                    
                    if len(risk_factors) > 0:
                        est_risk_factors_to_plot.append({})
                        for rf in risk_factors:
                            est_risk_factors_to_plot[-1][rf] = 0
                            for e in range(config["model"]["hyperparameters"]["num_rf_ensembles"]):
                                est_risk_factors_to_plot[-1][rf] += estimated_risk_factors[e][rf].detach().cpu() / config["model"]["hyperparameters"]["num_rf_ensembles"]

                    batch_fake_samples, _ = model.sample(config["dataloader"]["batch_size"], device=device)            
                    fake_samples_to_plot.append(batch_fake_samples.squeeze(1).detach().cpu())


            if config["logging"]["save_results_every_x_epochs"] != 0 and epoch % config["logging"]["save_results_every_x_epochs"] == 0:
                fig = figure.Figure(figsize=(config["logging"]["reconstructed_sample_fig_size"], 3 * config["logging"]["reconstructed_sample_fig_size"]))
                axs = fig.subplots(1, 3)

                # Inputs
                img = create_canvas(mris_to_plot, config["datasets"]["internal_train"]["input_shape"], config["dataloader"]["batch_size"], risk_factors_to_plot)
                axs[0].set_title("Epoch #{}, Test #{}, UKBB Samples".format(epoch, i_test))
                axs[0].axis("off")
                axs[0].imshow(img, cmap="gray")

                # Reconstructions
                img = create_canvas(recons_to_plot, config["datasets"]["internal_train"]["input_shape"], config["dataloader"]["batch_size"], est_risk_factors_to_plot)
                axs[1].set_title("Epoch #{}, Test #{}, Reconstructions".format(epoch, i_test))
                axs[1].axis("off")
                axs[1].imshow(img, cmap="gray")
                number_of_samples = len(recons_to_plot) * config["dataloader"]["batch_size"]

                # Samples
                img = create_canvas(fake_samples_to_plot, config["datasets"]["internal_train"]["input_shape"], config["dataloader"]["batch_size"])#, fake_risk_factors_to_plot)
                axs[2].set_title("Epoch #{}, Test #{}, Fake Samples".format(epoch, i_test))
                axs[2].axis("off")
                axs[2].imshow(img, cmap="gray")

                save_path = config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/results/" + "figure.e" + str(epoch) + ".test_" + str(i_test) + ".png"
                fig.savefig(save_path, bbox_inches="tight")


                f = open(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/errors/" + "errors.e" + str(epoch) + ".test_" + str(i_test) + ".txt", "w")
                for l in test_log[i_test]:
                    f.write(l + ": " + "{:.5f}".format(test_log[i_test][l]) + "\n")
                f.close()

                log["test_{}_loss".format(i_test)] = test_log[i_test]

        f = open(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/errors/" + "errors.e" + str(epoch) + ".train.txt", "w")
        for l in train_loss:
            f.write(l + ": " + "{:.5f}".format(train_loss[l]) + "\n")
        f.close()
        
        log["train_loss"] = train_loss
        log["lr"] = optimizer.param_groups[0]["lr"]
        log["beta"] = beta

        test_loss_total = []
        for i in range(len(validation_loaders)):
            test_loss_total.append(test_log[i]["total"])

        log["test_loss_total_mean"] = round(np.mean(test_loss_total), 5)
        log["test_loss_total_stdev"] = round(np.std(test_loss_total), 5)

        if config["logging"]["wandb"]["enabled"]:
            wandb.log(log, step=epoch)

    
    print("[Epoch: {}/{}]   Train Total Loss: {:.5f},   Test Total Loss: {:.5f}±{:.5f}".format(
        epoch, config["model"]["hyperparameters"]["num_epochs"], log["train_loss"]["total"], log["test_loss_total_mean"], log["test_loss_total_stdev"]))

    #scaler.step(scheduler)
    scheduler.step()


    if (config["logging"]["save_model_every_x_epochs"] != 0) and (epoch % config["logging"]["save_model_every_x_epochs"] == 0):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "current_epoch": epoch + 1,
            }, config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/" + "model.e{0}".format(epoch))

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "current_epoch": epoch + 1,
        }, config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/"  + "model.eLast")
