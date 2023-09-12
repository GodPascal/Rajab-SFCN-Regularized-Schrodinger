import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import decomposition
from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import cross_val_score

import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay

from config import config
import init
from statistics import mean, stdev

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
from model import RajabSFCNVAERegularizedLM2
from os import path
from data_test import MRIDataset



epoch = "120"
k = 5

rf_to_predict = "age"
darq_thr = 0.75
filter_dict = {"dx": {"type": "str", "values": ["cn"]}, "field_strength": {"type": "str", "values": ["1.5", "3.0"]}, "viscode": {"type": "str", "values": ["bl"]}, "darq": {"type": "float", "op": "gt", "thr": darq_thr}}
#filter_dict = {"dx": {"type": "str", "values": ["cn"]}, "viscode": {"type" : "str", "values": ["bl"]}}
config["logging"]["project_name"] = "Peng-Rajab"
config["logging"]["run_id"] = "ursula-23.08.29-bt40-bs4"
config["model"]["resume_training"] = True
config = init.parse_config(config, readonly=True)
config["dataloader"]["num_workers"] = 1
config["dataloader"]["batch_size"] = 1
config["datasets"]["internal_test"]["num_train_folds"] = 1
config["datasets"]["internal_test"]["train_folds"] = [0]
config["datasets"]["internal_test"]["num_subjects"] = 1
config["datasets"]["internal_test"]["path"] = "/export01/data/rrajabli/ukbb/"
config["datasets"]["internal_test"]["metadata_filepath"] = "/export01/data/rrajabli/ukbb/ukbb49140_07.28_darq_050_only_age_42.npz"
config["datasets"]["external_test"]["path"] = "/export02/data/rrajabli/adni/"
config["datasets"]["external_test"]["metadata_filepath"] = "/export02/data/rrajabli/adni/RADNIMERGE.06.14.npz"

affine_transforms_list = ["scale_x", "scale_y", "scale_z",
                     "rotation_x", "rotation_y", "rotation_z",
                     "translation_x", "translation_y", "translation_z"]
device = torch.device(config["misc"]["device"])
model = RajabSFCNVAERegularizedLM2(config["datasets"]["internal_train"]["input_shape"],
    config["dataloader"]["batch_size"],
    config["model"]["hyperparameters"]["dropout_last_layer"],
    risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"],
    affine_transforms=affine_transforms_list,
    variatioanl_bottleneck_size=config["model"]["hyperparameters"]["num_latent_factors"]).to(device)
# model = RajabSFCNVAERegularizedLM2(config["datasets"]["internal_train"]["input_shape"],
#     config["dataloader"]["batch_size"],
#     config["model"]["hyperparameters"]["dropout_last_layer"],
#     risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"],
#     last_layer_size=config["model"]["hyperparameters"]["num_latent_factors"]).to(device)

print(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/"  + "model.e" + str(epoch))
if path.exists(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/"  + "model.e" + str(epoch)) == False:
    print("[ERR] No model found to resume from!")
    exit()
last_state = torch.load(config["logging"]["path"] + config["logging"]["project_name"] + "/" + config["model"]["name"] + "_" + config["logging"]["run_id"] + "/saved_models/"  + "model.e" + str(epoch))
model.load_state_dict(last_state["model"])
del last_state
print("[INF] The model was loaded successfully!")
model.eval()

########### MANUAL INTERNAL TEST ############

data = MRIDataset(config["datasets"]["internal_test"]["path"],
        config["datasets"]["internal_test"]["metadata_filepath"],
        num_subjects=config["datasets"]["internal_test"]["num_subjects"],
        k=config["datasets"]["internal_test"]["num_train_folds"],
        fold=config["datasets"]["internal_test"]["train_folds"],
        risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"],
        #filter_dict=filter_dict,
        test_mode=True)

prds = []
for test_number in range(20):

    data_loader = torch.utils.data.DataLoader(data, batch_size=config["dataloader"]["batch_size"], num_workers=config["dataloader"]["num_workers"], shuffle=False, drop_last=True)

    with torch.no_grad():
        ids = []
        y = []
        y_hat = []
        features = []
        for idx, validation_batch in enumerate(data_loader):

            t1w = validation_batch["t1w"].to(device)
            risk_factors = {}
            for rf in config["model"]["hyperparameters"]["regularizers"]["risk_factors"]:
                risk_factors[rf] = validation_batch[rf].to(device)
            affine_transforms = {}
            for transform in affine_transforms_list:
                affine_transforms[transform] = validation_batch[transform].to(device)
            
            estimated_risk_factors, estimated_affine_transforms, recon, _, mu, logvar = model(t1w, affine_transforms)
            
            ids += validation_batch["id"]
            y += (validation_batch[rf_to_predict] * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["stdev"] + config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["mean"]).tolist()
            y_hat += (estimated_risk_factors[rf_to_predict].flatten() * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["stdev"] + config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["mean"]).tolist()
            features += [mu[0,:].tolist()]

            #print(f"{idx}/{len(data_loader)}", end="\r")

            print(f"Test #{test_number}- Y_hat: {y_hat[0]:.1f}")
            prds.append(y_hat[0])

print(mean(prds), stdev(prds))
exit()

print("")
print("EXTERNAL TEST DATA:")
print("")
print("\tNumber of Samples: " + str(len(data_loader) * config["dataloader"]["batch_size"]))
print("")
print("\tTest - Regulurizer Prediction:")
print(f"\t\tY: {y[0]:.1f}")
print(f"\t\tY_hat: {y_hat[0]:.1f}")
exit()

############### INTERNAL TEST ###############

# data = MRIDataset(config["datasets"]["internal_test"]["path"],
#         config["datasets"]["internal_test"]["metadata_filepath"],
#         num_subjects=config["datasets"]["internal_test"]["num_subjects"],
#         k=config["datasets"]["internal_test"]["num_folds"],
#         fold=config["datasets"]["internal_test"]["folds"],
#         risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"],
#         test_mode=True)
# data_loader = torch.utils.data.DataLoader(data, batch_size=config["dataloader"]["batch_size"], num_workers=config["dataloader"]["num_workers"], shuffle=False, drop_last=True)

# with torch.no_grad():
#     y = []
#     y_hat = []
#     for idx, validation_batch in enumerate(data_loader):

#         t1w = validation_batch["t1w"].to(device)
        
#         estimated_risk_factors, _, _, _, _, _ = model(t1w)

#         y += (validation_batch[rf_to_predict] * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["stdev"] + config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["mean"]).tolist()
#         y_hat += (estimated_risk_factors[rf_to_predict].flatten() * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["stdev"] + config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["mean"]).tolist()

#         print(f"{idx}/{len(data_loader)}", end="\r")

# print("")
# print("INTERNAL TEST DATA:")
# print("")
# print("\tNumber of Samples: " + str(len(data_loader) * config["dataloader"]["batch_size"]))
# print("")
# print("\tTest - Regulurizer Prediction:")
# print(f"\t\tMin: {np.min(y):.1f}")
# print(f"\t\tMax: {np.max(y):.1f}")
# print(f"\t\tRange: {np.mean(y):.1f} ± {np.std(y):.2f}")
# print(f"\t\tMAE: {np.mean(np.abs(np.subtract(y, y_hat))):.3f}")
# print(f"\t\tR^2: {np.corrcoef(y, y_hat)[0, 1] ** 2:.3f}")

# rf_df = pd.DataFrame({rf_to_predict: y, "Predicted Value": np.array(y_hat)})      
# plot = sns.jointplot(data=rf_df, x=rf_to_predict, y="Predicted Value", kind="reg")
# r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])
# print(f"\t\tpValue: {p:.5f}")
# sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value",
#             legend="full",
#             ax=plot.ax_joint)
# plt.show()

############### EXTERNAL TEST ###############

data = MRIDataset(config["datasets"]["external_test"]["path"],
        config["datasets"]["external_test"]["metadata_filepath"],
        num_subjects=config["datasets"]["internal_test"]["num_subjects"],
        k=config["datasets"]["external_test"]["num_train_folds"],
        fold=config["datasets"]["external_test"]["train_folds"],
        risk_factors=config["model"]["hyperparameters"]["regularizers"]["risk_factors"],
        filter_dict=filter_dict,
        test_mode=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=config["dataloader"]["batch_size"], num_workers=config["dataloader"]["num_workers"], shuffle=False, drop_last=True)

with torch.no_grad():
    ids = []
    y = []
    y_hat = []
    features = []
    for idx, validation_batch in enumerate(data_loader):

        t1w = validation_batch["t1w"].to(device)
        
        estimated_risk_factors, _, _, mu, _ = model(t1w)

        ids += validation_batch["ptid"]
        y += (validation_batch[rf_to_predict] * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["stdev"] + config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["mean"]).tolist()
        y_hat += (estimated_risk_factors[rf_to_predict].flatten() * config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["stdev"] + config["model"]["hyperparameters"]["regularizers"]["risk_factors"][rf_to_predict]["mean"]).tolist()
        features += [mu[0,:].tolist()]

        print(f"{idx}/{len(data_loader)}", end="\r")

print("")
print("EXTERNAL TEST DATA:")
print("")
print("\tNumber of Samples: " + str(len(data_loader) * config["dataloader"]["batch_size"]))
print("")
print("\tTest - Regulurizer Prediction:")
print(f"\t\tMin: {np.min(y):.1f}")
print(f"\t\tMax: {np.max(y):.1f}")
print(f"\t\tRange: {np.mean(y):.1f} ± {np.std(y):.2f}")
print(f"\t\tMAE: {np.mean(np.abs(np.subtract(y, y_hat))):.3f}")
print(f"\t\tR^2: {np.corrcoef(y, y_hat)[0, 1] ** 2:.3f}")

rf_df = pd.DataFrame({rf_to_predict: y, "Predicted Value": np.array(y_hat)})
r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])

fig, axes = plt.subplots(3, 3, gridspec_kw={"width_ratios": (.8, .1, .1), "height_ratios": (.1, .1, .8)})
fig.set_size_inches(12, 12)
sns.histplot(data=rf_df, x=rf_to_predict, ax=axes[0, 0], bins=int(max(rf_df[rf_to_predict]))-int(min(rf_df[rf_to_predict]))+1, kde=True)
sns.boxplot(data=rf_df, x=rf_to_predict, ax=axes[1, 0])
sns.histplot(data=rf_df, y="Predicted Value", ax=axes[2, 2], bins=int(max(rf_df["Predicted Value"]))-int(min(rf_df["Predicted Value"]))+1, kde=True)
sns.boxplot(data=rf_df, y="Predicted Value", ax=axes[2, 1])
sns.kdeplot(data=rf_df, x=rf_to_predict, y="Predicted Value", ax=axes[2, 0])
sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value", alpha=.5, ax=axes[2, 0], legend=False)
axes[0, 0].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[0, 1].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[0, 2].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[1, 0].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[1, 1].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[1, 2].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[2, 0].set(xlabel='', ylabel='', xbound=[50, 100], ybound=[50, 100])
axes[2, 1].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[2, 2].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)

plt.show()


############### EXTERNAL TEST - Linear Regression ###############

regr = Ridge()
gkf = GroupKFold(n_splits=k)

XX = np.array(features)
yy = np.array(y)

maes = []
r2s = []
best_ukbb_regr = None

x_reg_plot = np.array([])
y_reg_plot = np.array([])

np.random.seed(42)
for train_ix, test_ix in gkf.split(XX, yy, groups=ids):
    X_train = XX[train_ix]
    X_test = XX[test_ix]
    y_train = yy[train_ix]
    y_test = yy[test_ix]
    
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)

    mae = np.mean(abs(y_predict - y_test))
    r2 = np.corrcoef([y_predict, y_test])[0][1] ** 2
    if len(r2s) == 0 or r2 > np.max(r2s):
    #if len(maes) == 0 or mae > np.min(maes):
        best_ukbb_regr = regr
    maes.append(mae)
    r2s.append(r2)

    x_reg_plot = np.concatenate([x_reg_plot, y_test])
    y_reg_plot = np.concatenate([y_reg_plot, y_predict])

print("")
print("EXTERNAL TEST DATA - Linear Regression:")
print("")
print("\tNumber of Samples: " + str(len(data_loader) * config["dataloader"]["batch_size"]))
print("")
print("\tTest - Regulurizer Prediction:")
print(f"\t\tMin: {np.min(y):.1f}")
print(f"\t\tMax: {np.max(y):.1f}")
print(f"\t\tRange: {np.mean(y):.1f} ± {np.std(y):.2f}")
print(f"\t\tMAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")
print(f"\t\tR^2: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

rf_df = pd.DataFrame({rf_to_predict: x_reg_plot, "Predicted Value": np.array(y_reg_plot)})      
r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])

fig, axes = plt.subplots(3, 3, gridspec_kw={"width_ratios": (.8, .1, .1), "height_ratios": (.1, .1, .8)})
fig.set_size_inches(12, 12)
sns.histplot(data=rf_df, x=rf_to_predict, ax=axes[0, 0], bins=int(max(rf_df[rf_to_predict]))-int(min(rf_df[rf_to_predict]))+1, kde=True)
sns.boxplot(data=rf_df, x=rf_to_predict, ax=axes[1, 0])
sns.histplot(data=rf_df, y="Predicted Value", ax=axes[2, 2], bins=int(max(rf_df["Predicted Value"]))-int(min(rf_df["Predicted Value"]))+1, kde=True)
sns.boxplot(data=rf_df, y="Predicted Value", ax=axes[2, 1])
sns.kdeplot(data=rf_df, x=rf_to_predict, y="Predicted Value", ax=axes[2, 0])
sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value", alpha=.5, ax=axes[2, 0], legend=False)
axes[0, 0].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[0, 1].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[0, 2].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[1, 0].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[1, 1].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[1, 2].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[2, 0].set(xlabel='', ylabel='', xbound=[50, 100], ybound=[50, 100])
axes[2, 1].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)
axes[2, 2].set(xlabel='', xticks=[], ylabel='', yticks=[], frame_on=False)

plt.show()


exit()

np.random.seed(42)
gkf = GroupKFold(n_splits=k)

XX = np.array(X)
yy = np.array(y)
if is_there_reg:
    rr = np.array(reg)

############### TEST #1 - UKBB REG ###############
if is_there_reg:
    maes = []
    r2s = []
    
    x_reg_plot = np.array([])
    y_reg_plot = np.array([])

    np.random.seed(42)
    for train_ix, test_ix in gkf.split(X, y, groups=ids):
        y_test = yy[test_ix]
        y_reg = rr[test_ix]
        
        # for reg_idx in range(len(y_reg)):
        #     min = 100
        #     for y_test_value in y_test:
        #         if abs(rr[test_ix][reg_idx] - y_test_value) < min:
        #             y_reg[reg_idx] = y_test_value
        #             min = abs(rr[test_ix][reg_idx] - y_test_value)
        
        mae = np.mean(abs(y_reg - y_test))
        maes.append(mae)
        r2 = np.corrcoef([y_reg, y_test])[0][1] ** 2
        r2s.append(r2)

        x_reg_plot = np.concatenate([x_reg_plot, y_test])
        y_reg_plot = np.concatenate([y_reg_plot, y_reg])

    print("")
    print("\tTest - Regulurizer Prediction:")
    print(f"\t\tMin: {np.min(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
    print(f"\t\tMax: {np.max(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
    print(f"\t\tRange: {np.mean(y) * rf_to_predict_stdev + rf_to_predict_mean:.1f} ± {np.std(y) * rf_to_predict_stdev:.2f}")
    print(f"\t\tMAE: {np.mean(maes) * rf_to_predict_stdev:.3f} ± {np.std(maes) * rf_to_predict_stdev:.3f}")
    print(f"\t\tR^2: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    rf_df = pd.DataFrame({rf_to_predict: x_reg_plot, "Predicted Value": np.array(y_reg_plot)})      
    plot = sns.jointplot(data=rf_df, x=rf_to_predict, y="Predicted Value", kind="reg")
    r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])
    print(f"\t\tpValue: {p:.5f}")
    sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value",
                legend="full",
                ax=plot.ax_joint)
    plt.show()


############### TEST #2 - UKBB LINEAR ###############
regr = LinearRegression()
best_ukbb_regr = None

maes = []
r2s = []

x_reg_plot = np.array([])
y_reg_plot = np.array([])

np.random.seed(42)
for train_ix, test_ix in gkf.split(X, y, groups=ids):
    X_train = XX[train_ix]
    X_test = XX[test_ix]
    y_train = yy[train_ix]
    y_test = yy[test_ix]
    
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)

    mae = np.mean(abs(y_predict - y_test))
    r2 = np.corrcoef([y_predict, y_test])[0][1] ** 2
    if len(r2s) == 0 or r2 > np.max(r2s):
    #if len(maes) == 0 or mae > np.min(maes):
        best_ukbb_regr = regr
    maes.append(mae)
    r2s.append(r2)

    x_reg_plot = np.concatenate([x_reg_plot, y_test])
    y_reg_plot = np.concatenate([y_reg_plot, y_predict])

print("")
print("\tTest - Linear Model Prediction:")
print(f"\t\tMin: {np.min(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
print(f"\t\tMax: {np.max(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
print(f"\t\tRange: {np.mean(y) * rf_to_predict_stdev + rf_to_predict_mean:.1f} ± {np.std(y) * rf_to_predict_stdev:.2f}")
print(f"\t\tMAE: {np.mean(maes) * rf_to_predict_stdev:.3f} ± {np.std(maes) * rf_to_predict_stdev:.3f}")
print(f"\t\tR^2: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

rf_df = pd.DataFrame({rf_to_predict: x_reg_plot, "Predicted Value": np.array(y_reg_plot)})      
plot = sns.jointplot(data=rf_df, x=rf_to_predict, y="Predicted Value", kind="reg")
r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])
print(f"\t\tpValue: {p:.5f}")
sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value",
            legend="full",
            ax=plot.ax_joint)
plt.show()


#############################################
#############################################
#############################################


ids = []
X = []
reg = []
y = []

adni_metadata = np.load(adni_metadata_file_path, allow_pickle=True)
adni_metadata = adni_metadata["data"]

with open(adni_test_data_csv_path, "r") as f:
    csvreader = csv.reader(f, delimiter=",")
    counter = 0
    for row in csvreader:
        sample_included = False
        for adni_sample in adni_metadata:
            if adni_sample["subdir"] == row[0]:
                sample_included = True
                if float(adni_sample["darq"]) < darq_thr:
                    sample_included = False
                    break
                if adni_sample["dx"] not in dx_included:
                    sample_included = False
                    break
                if adni_sample["field_strength"] not in field_strength_included:
                    sample_included = False
                    break
                # if float(adni_sample["age"]) < 50 or float(adni_sample["age"]) > 80:
                #     sample_included = False
                #     break
                if sample_included:
                    y.append((float(adni_sample[rf_to_predict]) - rf_to_predict_mean) / rf_to_predict_stdev)
                break
        if not(sample_included):
            continue

        counter += 1
        
        # if counter == 1440:
        #     break

        if row[0][3] == "1":
            ids.append(row[0][7:17])
        else:
            ids.append(row[0][6:16])

        new_row = []
        for i in range(latent_features_index_start_in_csv, latent_features_index_end_in_csv):
            if i % 2 == latent_features_index_start_in_csv % 2:
                new_row.append(float(row[i]))
        X.append(new_row)

        if is_there_reg:
            reg.append(float(row[rf_reg_index_in_csv]))

X = np.array(X)#StandardScaler().fit_transform(X)
if is_there_reg:
    reg = np.array(reg)
y = np.array(y)

print("")
print("ADNI TEST DATA:")
print("")
print("\tNumber of Samples: " + str(len(X)))
print("\tNumber of Features: " + str(len(X[0])))

np.random.seed(42)
gkf = GroupKFold(n_splits=k)

XX = np.array(X)
yy = np.array(y)
if is_there_reg:
    rr = np.array(reg)

############### TEST #3 - ADNI REG ###############
if is_there_reg:
    maes = []
    r2s = []

    x_reg_plot = np.array([])
    y_reg_plot = np.array([])

    np.random.seed(42)
    for train_ix, test_ix in gkf.split(X, y, groups=ids):
        y_test = yy[test_ix]
        y_reg = rr[test_ix]
        
        mae = np.mean(abs(y_reg - y_test))
        maes.append(mae)
        r2 = np.corrcoef([y_reg, y_test])[0][1] ** 2
        r2s.append(r2)

        x_reg_plot = np.concatenate([x_reg_plot, y_test])
        y_reg_plot = np.concatenate([y_reg_plot, y_reg])

    print("")
    print("\tTest - Regulurizer Prediction:")
    print(f"\t\tMin: {np.min(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
    print(f"\t\tMax: {np.max(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
    print(f"\t\tRange: {np.mean(y) * rf_to_predict_stdev + rf_to_predict_mean:.1f} ± {np.std(y) * rf_to_predict_stdev:.2f}")
    print(f"\t\tMAE: {np.mean(maes) * rf_to_predict_stdev:.3f} ± {np.std(maes) * rf_to_predict_stdev:.3f}")
    print(f"\t\tR^2: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    
    rf_df = pd.DataFrame({rf_to_predict: x_reg_plot, "Predicted Value": np.array(y_reg_plot)})      
    plot = sns.jointplot(data=rf_df, x=rf_to_predict, y="Predicted Value", kind="reg")
    r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])
    print(f"\t\tpValue: {p:.5f}")
    sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value",
                legend="full",
                ax=plot.ax_joint)
    plt.show()


############### TEST #4 - ADNI W UKBB LINEAR ###############
maes = []
r2s = []

x_reg_plot = np.array([])
y_reg_plot = np.array([])

np.random.seed(42)
for train_ix, test_ix in gkf.split(X, y, groups=ids):
    X_train = XX[train_ix]
    X_test = XX[test_ix]
    y_train = yy[train_ix]
    y_test = yy[test_ix]
    
    y_predict = best_ukbb_regr.predict(X_test)
    mae = np.mean(abs(y_predict - y_test))
    maes.append(mae)
    r2 = np.corrcoef([y_predict, y_test])[0][1] ** 2
    r2s.append(r2)

    x_reg_plot = np.concatenate([x_reg_plot, y_test])
    y_reg_plot = np.concatenate([y_reg_plot, y_predict])

print("")
print("\tTest - Linear Model [Fitted on UKBB] Prediction:")
print(f"\t\tMin: {np.min(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
print(f"\t\tMax: {np.max(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
print(f"\t\tRange: {np.mean(y) * rf_to_predict_stdev + rf_to_predict_mean:.1f} ± {np.std(y) * rf_to_predict_stdev:.2f}")
print(f"\t\tMAE: {np.mean(maes) * rf_to_predict_stdev:.3f} ± {np.std(maes) * rf_to_predict_stdev:.3f}")
print(f"\t\tR^2: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

rf_df = pd.DataFrame({rf_to_predict: x_reg_plot, "Predicted Value": np.array(y_reg_plot)})      
plot = sns.jointplot(data=rf_df, x=rf_to_predict, y="Predicted Value", kind="reg")
r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])
print(f"\t\tpValue: {p:.5f}")
sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value",
            legend="full",
            ax=plot.ax_joint)
plt.show()


############### TEST #5 - ADNI LINEAR ###############
regr = LinearRegression()

maes = []
r2s = []

x_reg_plot = np.array([])
y_reg_plot = np.array([])

np.random.seed(42)
for train_ix, test_ix in gkf.split(X, y, groups=ids):
    X_train = XX[train_ix]
    X_test = XX[test_ix]
    y_train = yy[train_ix]
    y_test = yy[test_ix]
    
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    mae = np.mean(abs(y_predict - y_test))
    maes.append(mae)
    r2 = np.corrcoef([y_predict, y_test])[0][1] ** 2
    r2s.append(r2)
    
    x_reg_plot = np.concatenate([x_reg_plot, y_test])
    y_reg_plot = np.concatenate([y_reg_plot, y_predict])

print("")
print("\tTest - Linear Model [Fitted on ADNI] Prediction:")
print(f"\t\tMin: {np.min(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
print(f"\t\tMax: {np.max(y * rf_to_predict_stdev + rf_to_predict_mean):.1f}")
print(f"\t\tRange: {np.mean(y) * rf_to_predict_stdev + rf_to_predict_mean:.1f} ± {np.std(y) * rf_to_predict_stdev:.2f}")
print(f"\t\tMAE: {np.mean(maes) * rf_to_predict_stdev:.3f} ± {np.std(maes) * rf_to_predict_stdev:.3f}")
print(f"\t\tR^2: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

rf_df = pd.DataFrame({rf_to_predict: x_reg_plot, "Predicted Value": np.array(y_reg_plot)})      
plot = sns.jointplot(data=rf_df, x=rf_to_predict, y="Predicted Value", kind="reg")
r, p = stats.pearsonr(rf_df[rf_to_predict], rf_df["Predicted Value"])
print(f"\t\tpValue: {p:.5f}")
sns.scatterplot(data=rf_df, x=rf_to_predict, y="Predicted Value",
            legend="full",
            ax=plot.ax_joint)
print("")
plt.show()
