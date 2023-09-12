## There is no uptight validator for this one, change it precautiosly!

on_computecanada = False

config = {
    
    "logging": {
        
        "path": "./logs/",
        "project_name": "Peng-Rajab",
        "run_id": "", # config parser will take care of this
        "save_model_every_x_epochs": 1,
        "save_results_every_x_epochs": 1,
        "num_reconstructed_samples_to_save": 16,
        "reconstructed_sample_fig_size": 20,

        "wandb": {

            "enabled": False,
            "entity": "mcgill",
            "project_name": "", # config parser will take care of this
            "run_id": "", # config parser will take care of this

        }

    },

    "datasets": {

        "internal_train": {

            "path": "/home/rrajabli/scratch/ukbb/" if on_computecanada else "/export01/data/rrajabli/ukbb/",
            "input_shape": (160, 192, 160),
            "metadata_filepath": "/home/rrajabli/scratch/ukbb/ukbb49140_07.28_darq_050_only_age_42.npz" if on_computecanada else "/export01/data/rrajabli/ukbb/ukbb49140_07.28_darq_050_only_age_42.npz",
            "num_subjects": -1, # all subjects in the set
            "num_train_folds": 20,
            "train_folds": [*range(17)],

        },

        "internal_validation": {

            "path": "/home/rrajabli/scratch/ukbb/" if on_computecanada else "/export01/data/rrajabli/ukbb/",
            "input_shape": (160, 192, 160),
            "metadata_filepath": "/home/rrajabli/scratch/ukbb/ukbb49140_07.28_darq_050_only_age_42.npz" if on_computecanada else "/export01/data/rrajabli/ukbb/ukbb49140_07.28_darq_050_only_age_42.npz",
            "num_subjects": -1, # all subjects in the set
            "num_train_folds": 20,
            "train_folds": [17, 18],

        },

        "internal_test": {

            "path": "/home/rrajabli/scratch/ukbb/" if on_computecanada else "/export01/data/rrajabli/ukbb/",
            "input_shape": (160, 192, 160),
            "metadata_filepath": "/home/rrajabli/scratch/ukbb/ukbb49140_07.28_darq_050_only_age_42.npz" if on_computecanada else "/export01/data/rrajabli/ukbb/ukbb49140_07.28_darq_050_only_age_42.npz",
            "num_subjects": -1, # all subjects in the set
            "num_train_folds": 20,
            "train_folds": [19],

        },

        "external_test": {

            "path": "/home/rrajabli/scratch/adni/" if on_computecanada else "/export01/data/rrajabli/adni/",
            "input_shape": (160, 192, 160),
            "metadata_filepath": "/home/rrajabli/scratch/adni/RADNIMERGE.06.14.npz" if on_computecanada else "/export01/data/rrajabli/adni/RADNIMERGE.06.14.npz",
            "num_subjects": -1, # all subjects in the set
            "num_train_folds": 1,
            "train_folds": [0],

        }

    },

    "dataloader": {
        
        "batch_size": 8 if on_computecanada else 4,
        "num_workers": 36 if on_computecanada else 8 # Do not change this one, unless you know what you are doing!

    },

    "model": {

        "name": "RegularizedRajabSFCN",
        "resume_training": False,

        "hyperparameters": {

            "num_epochs": 128,
            "l2_weight_decay": 1e-3,
            "dropout_last_layer": 0.5,
            "lr": 1e-2,
            "lr_decay_step_size": 30,
            "lr_decay_gamma": 0.3,
            "gradient_clipping": False,
            "gradient_min": -10.0,
            "gradient_max": 10.0,

            "num_latent_factors": 40,
            "num_rf_ensembles": 100,
            "beta_max": 1e+1,
            "beta_normalizing": True,
            "beta_num_cycles": 2,
            "beta_r": 0.5,
            "vae_error_coeff": 1e-6, ## Reconstruction error coefficient            

            "regularizers": {
            
                "risk_factors": {
                    
                    # "sex": {
                    #     "type": "discrete",
                    #     "labels": ["female", "male"],
                    #     "class_balance": [], # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #     "balance_error_coeff": False
                    # },

                    # "lonliness": {
                    #     "type": "discrete",
                    #     "labels": ["no", "yes"],
                    #     "class_balance": [], # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #    "balance_error_coeff": False
                    # },

                    # "depressed_in_last_2w": {
                    #     "type": "discrete",
                    #     "labels": ["no", "yes"],
                    #     "class_balance": [], # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #     "balance_error_coeff": False
                    # },

                    # "hearing_difficulty": {
                    #     "type": "discrete",
                    #     "labels": ["no", "yes"],
                    #     "class_balance": [], # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #     "balance_error_coeff": False
                    # },

                    # "diabetes": {
                    #     "type": "discrete",
                    #     "labels": ["no", "yes"],
                    #     "class_balance": [], # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #     "balance_error_coeff": False
                    # },

                    # "smoking_status": {
                    #     "type": "discrete",
                    #     "labels": ["never", "previous", "current"],
                    #     "class_balance": [], # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #     "balance_error_coeff": False
                    # },

                    "age": {
                        "type": "continuous",
                        "mean": -1, # config parser will take care of this
                        "stdev": -1, # config parser will take care of this
                        "error_coeff": 0.01,#160 * 192 * 160,
                        "z_score_tuned_error_coeff": False,
                        "skip_outliers": False
                    },

                    # "bmi": {
                    #     "type": "continuous",
                    #     "mean": -1, # config parser will take care of this
                    #     "stdev": -1, # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #     "z_score_tuned_error_coeff": True,
                    #     "skip_outliers": True
                    # },

                    # "wmh": {
                    #     "type": "continuous",
                    #     "mean": -1, # config parser will take care of this
                    #     "stdev": -1, # config parser will take care of this
                    #     "error_coeff": 1.0,
                    #     "z_score_tuned_error_coeff": True,
                    #     "skip_outliers": True
                    # },

                }

            }
        
        }
        
    },

    "misc": {

        "device": "cuda",
        "python_random_seed": 42,
        "torch_random_seed": 42

    }

}
