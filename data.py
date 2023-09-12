import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from torchio import RandomAffine, RandomBiasField, RandomMotion, RandomNoise, RescaleIntensity, RandomGamma, RandomBlur
from torchio import Compose, Subject, ScalarImage, LabelMap

import matplotlib.pyplot as plt


class MRIDataset(Dataset):

    def __init__(self, path, metadata_filepath, num_subjects=-1, risk_factors=[], k=1, fold=0, filter_dict={}, test_mode=False):
        self.path = path
        self.risk_factors = risk_factors
        self.test_mode = test_mode

        self.metadata = np.load(metadata_filepath, allow_pickle=True)
        self.metadata = self.metadata["data"]
        if num_subjects > 0:
            self.metadata = self.metadata[:num_subjects]
        
        indices = []
        for i in range(len(self.metadata)):
            skip = False
            for f in filter_dict:
                if filter_dict[f]["type"] == "str":
                    if self.metadata[i][f] not in filter_dict[f]["values"]:
                        skip = True
                        break
                if filter_dict[f]["type"] == "float":
                    if filter_dict[f]["op"] == "gt":
                        if float(self.metadata[i][f]) < filter_dict[f]["thr"]:
                            skip = True
                            break
                    if filter_dict[f]["op"] == "lt":
                        if float(self.metadata[i][f]) > filter_dict[f]["thr"]:
                            skip = True
                            break
            if skip:
                continue
            if type(fold) == int:
                if (i % k == fold):
                    indices.append(i)
            elif type(fold) == list:
                if (i % k in fold):
                    indices.append(i)
        self.metadata = self.metadata[indices]

        for idx in range(len(self.metadata)):
            for rf in self.risk_factors:
                if self.risk_factors[rf]["type"] == "discrete":
                    one_hot_label = np.array([0.0] * len(self.risk_factors[rf]["labels"]))
                    one_hot_label[self.risk_factors[rf]["labels"].index(self.metadata[idx][rf])] = 1.0
                    self.metadata[idx][rf] = torch.tensor(one_hot_label, dtype=torch.float)
                elif self.risk_factors[rf]["type"] == "continuous":
                    self.metadata[idx][rf] = torch.tensor((float(self.metadata[idx][rf]) - self.risk_factors[rf]["mean"]) / self.risk_factors[rf]["stdev"], dtype=torch.float)

    
    def __len__(self):
        return len(self.metadata)


    def __load_subject_data__(self, idx):
        
        t1w = np.load(self.path + self.metadata[idx]["subdir"] + "/" + "t1w_256_tight.npz")
        t1w = torch.from_numpy(t1w["data"])
        # t1w = torch.nn.functional.pad(t1w, (16, 16, 0, 0, 16, 16))
        # with open(self.path + self.metadata[idx]["subdir"] + "/" + "t1w_256_tight.pkl", "rb") as f:
        #     t1w = pickle.load(f)
        # t1w = torch.from_numpy(t1w)

        t1w = t1w.unsqueeze(0)

        t1w_mask = np.load(self.path + self.metadata[idx]["subdir"] + "/" + "t1w_256_mask_tight.npz")
        t1w_mask = torch.from_numpy(t1w_mask["data"])
        # t1w_mask = torch.nn.functional.pad(t1w_mask, (16, 16, 0, 0, 16, 16))
        # with open(self.path + self.metadata[idx]["subdir"] + "/" + "t1w_256_mask_tight.pkl", "rb") as f:
        #     t1w_mask = pickle.load(f)
        # t1w_mask = torch.from_numpy(t1w_mask)

        t1w_mask = t1w_mask.unsqueeze(0)

        subject = Subject(t1w=ScalarImage(tensor=t1w),
                          t1w_mask=LabelMap(tensor=t1w_mask),
                          t1w_orig=ScalarImage(tensor=t1w.clone()),
                          t1w_orig_mask=ScalarImage(tensor=t1w_mask.clone()))

        ## This part was tuned by Louis
        transforms = []
        transforms.append(RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), exclude=["t1w_mask", "t1w_orig_mask"]))
        transforms.append(RandomAffine(scales=(.98, 1.02, .98, 1.02, .98, 1.02), degrees=(-9, 9, -9, 9, -9, 9), translation=(-4, 4, -4, 4, -4, 4)))
        if self.test_mode == False:
            transforms.append(RandomGamma((-0.3, 0.3), exclude=["t1w_mask", "t1w_orig", "t1w_orig_mask"]))
            if np.random.rand() < 0.1:
                transforms.append(RandomBiasField((0.0, 0.3), order=1, exclude=["t1w_mask", "t1w_orig", "t1w_orig_mask"]))
            transforms.append(RandomMotion(degrees=0, translation=(0.0, 3.0), num_transforms=np.random.randint(4)+1, exclude=["t1w_mask", "t1w_orig", "t1w_orig_mask"]))
            transforms.append(RandomBlur(std=[0.0, 0.2], exclude=["t1w_mask", "t1w_orig", "t1w_orig_mask"]))
            transforms.append(RandomNoise(mean=0, std=[0, 0.04], exclude=["t1w_mask", "t1w_orig", "t1w_orig_mask"]))
            transforms.append(RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), exclude=["t1w_mask", "t1w_orig_mask"]))
        final_transform = Compose(transforms)
        subject = final_transform(subject)

        t1w = subject.t1w.tensor.squeeze(0)
        t1w_mask = subject.t1w_mask.tensor.squeeze(0)
        t1w_orig = subject.t1w_orig.tensor.squeeze(0)
        t1w_orig_mask = subject.t1w_orig_mask.tensor.squeeze(0)
        # if self.test_mode == False:
        #     voxel_shift = np.random.randint(-2, 3)
        #     t1w = torch.roll(t1w, shifts=(voxel_shift, voxel_shift, voxel_shift), dims=(0, 1, 2))
        #     t1w_mask = torch.roll(t1w_mask, shifts=(voxel_shift, voxel_shift, voxel_shift), dims=(0, 1, 2))

        t1w = (t1w - torch.mean(t1w[t1w_mask != 0])) / torch.std(t1w[t1w_mask != 0])
        if torch.std(t1w[t1w_mask != 0]) < 1e-5:
            print(idx, "Blank MRI Detected!", self.metadata[idx]["subdir"])
            exit()
        t1w *= t1w_mask

        t1w_orig = (t1w_orig - torch.mean(t1w_orig[t1w_orig_mask != 0])) / torch.std(t1w_orig[t1w_orig_mask != 0])
        t1w_orig *= t1w_orig_mask

        # plt.subplot(1, 2, 1)
        # plt.imshow(t1w_orig[80, :, :], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.imshow(t1w[80, :, :], cmap="gray")
        # plt.show()
        
        sample = {"id": self.metadata[idx]["subdir"], "rnd": torch.rand(1, dtype=torch.float)}

    #     Compose(
    #         [RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method=None), Affine(scales=(0.9877943992614746, 1.0009671449661255, 0.9879487156867981), degrees=(0.06768479198217392, 1.5598551034927368, 0.6918566823005676), translation=(2.642444133758545, -3.9602794647216797, 0.8244872093200684), center=image, default_pad_value=minimum, image_interpolation=linear, label_interpolation=nearest, check_shape=True), Gamma(gamma={'t1w': [0.87787046043534]}), Motion(degrees={'t1w': array([[0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.]], dtype=float32)}, translation={'t1w': array([[2.8391342, 2.0117989, 2.2954922],
    #    [1.6049724, 1.2013228, 1.7960575],
    #    [1.0298445, 2.6642833, 1.6002842]], dtype=float32)}, times={'t1w': array([0.28752768, 0.43488127, 0.7981298 ], dtype=float32)}, image_interpolation={'t1w': 'linear'}), Blur(std={'t1w': tensor([0.1578, 0.1030, 0.0529])}), Noise(mean={'t1w': 0.0}, std={'t1w': 0.023999787867069244}, seed={'t1w': 2054056299}), RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), masking_method=None)])
        for transform in subject.get_composed_history():
            if transform.name == "Affine":
                sample["scale_x"] = (torch.tensor(transform.__dict__["scales"][0], dtype=torch.float) - 1.0) / np.sqrt((1.02 - 0.98) ** 2 / 12)
                sample["scale_y"] = (torch.tensor(transform.__dict__["scales"][1], dtype=torch.float) - 1.0) / np.sqrt((1.02 - 0.98) ** 2 / 12)
                sample["scale_z"] = (torch.tensor(transform.__dict__["scales"][2], dtype=torch.float) - 1.0) / np.sqrt((1.02 - 0.98) ** 2 / 12)
                sample["rotation_x"] = torch.tensor(transform.__dict__["degrees"][0], dtype=torch.float) / np.sqrt((9 - -9) ** 2 / 12)
                sample["rotation_y"] = torch.tensor(transform.__dict__["degrees"][1], dtype=torch.float) / np.sqrt((9 - -9) ** 2 / 12)
                sample["rotation_z"] = torch.tensor(transform.__dict__["degrees"][2], dtype=torch.float) / np.sqrt((9 - -9) ** 2 / 12)
                sample["translation_x"] = torch.tensor(transform.__dict__["translation"][0], dtype=torch.float) / np.sqrt((4 - -4) ** 2 / 12)
                sample["translation_y"] = torch.tensor(transform.__dict__["translation"][1], dtype=torch.float) / np.sqrt((4 - -4) ** 2 / 12)
                sample["translation_z"] = torch.tensor(transform.__dict__["translation"][2], dtype=torch.float) / np.sqrt((4 - -4) ** 2 / 12)

        sample["t1w"] = t1w.unsqueeze(0)
        sample["t1w_mask"] = t1w_mask.unsqueeze(0)
        sample["t1w_orig"] = t1w_orig.unsqueeze(0)
        sample["t1w_orig_mask"] = t1w_orig_mask.unsqueeze(0)
        
        for rf in self.risk_factors:
            sample[rf] = self.metadata[idx][rf]

        return sample

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        return self.__load_subject_data__(idx)
