import torch
import torch.nn as nn
from numpy import prod
from torch import randn_like


class SFCN_Encoder_Block(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size=3, max_pooling=True):
        super(SFCN_Encoder_Block, self).__init__()

        self.conv       = nn.Conv3d(num_in_channels, num_out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm3d(num_out_channels)
        self.max_pool   = nn.MaxPool3d(2) if max_pooling else None
        self.relu       = nn.ReLU()
    
    def forward(self, x):
        x = self.batch_norm(self.conv(x))
        if self.max_pool is not None:
            x = self.max_pool(x)
        return self.relu(x)


class SFCN_Last_Block(nn.Module):
    def __init__(self, num_in_channels, feature_map_size, num_out_channels, kernel_size=1):
        super(SFCN_Last_Block, self).__init__()

        self.avg_pool   = nn.AvgPool3d(feature_map_size)
        self.conv       = nn.Conv3d(num_in_channels, num_out_channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(self.avg_pool(x))[:, :, 0, 0, 0] # Manual flattening
        return x


class ConvEncoder(nn.Module):

    def __init__(self, num_channels_for_each_layer=[1, 32, 64, 128, 256, 256, 64]):
        super(ConvEncoder, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(len(num_channels_for_each_layer) - 2):
            self.blocks.append(SFCN_Encoder_Block(num_in_channels=num_channels_for_each_layer[i], num_out_channels=num_channels_for_each_layer[i+1]))
        self.blocks.append(SFCN_Encoder_Block(num_in_channels=num_channels_for_each_layer[-2], num_out_channels=num_channels_for_each_layer[-1], kernel_size=1, max_pooling=False))
        
    def forward(self, x):
        for l in self.blocks:
            x = l(x)
        return x


class PredictorRF(nn.Module):

    def __init__(self, num_features, risk_factors={}):
        super(PredictorRF, self).__init__()

        self.predictors = nn.ModuleDict() 
        for rf in risk_factors:
            if risk_factors[rf]["type"] == "continuous":
                self.predictors.add_module(rf, nn.Linear(num_features, 1))
            if risk_factors[rf]["type"] == "discrete":
                self.predictors.add_module(rf, nn.Linear(num_features, len(risk_factors[rf]["labels"])))
        
    def forward(self, x):
        estimated_risk_factors = {}
        for rf in self.predictors:
            estimated_risk_factors[rf] = self.predictors[rf](x)
        return estimated_risk_factors
    

class PredictorAffine(nn.Module):

    def __init__(self, num_in_channels, feature_map_size, affine_transforms={}):
        super(PredictorAffine, self).__init__()

        self.predictors = nn.ModuleDict() 
        # for rf in risk_factors:
        #     if risk_factors[rf]["type"] == "continuous":
        #         self.predictors.add_module(rf, nn.Linear(num_features, 1))
        #     if risk_factors[rf]["type"] == "discrete":
        #         self.predictors.add_module(rf, nn.Linear(num_features, len(risk_factors[rf]["labels"])))
        for transform in affine_transforms:
            self.predictors.add_module(transform, SFCN_Last_Block(num_in_channels, feature_map_size, 1))
        
    def forward(self, x):
        estimated_risk_factors = {}
        for rf in self.predictors:
            estimated_risk_factors[rf] = self.predictors[rf](x)
        return estimated_risk_factors


class SFCN_Like_Decoder_Block(nn.Module):

    def __init__(self, num_in_channels, num_out_channels, kernel_size=3, deconv=True, last_layer=False):
        super(SFCN_Like_Decoder_Block, self).__init__()

        self.deconv     = nn.ConvTranspose3d(num_in_channels, num_in_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=2, output_padding=1) if deconv else None
        self.conv       = nn.Conv3d(num_in_channels, num_out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm3d(num_out_channels) if not(last_layer) else None
        self.relu       = nn.ReLU() if not(last_layer) else None
    
    def forward(self, x):
        if self.deconv is not None:
            x = self.deconv(x)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ConvDecoder(nn.Module):

    def __init__(self, feature_map_size, num_latent_factors=40, num_channels_for_each_dec_layer=[64, 256, 256, 128, 64, 32, 1]):
        super(ConvDecoder, self).__init__()

        self.feature_map_size = feature_map_size
        self.num_dec_channels_first_layer = num_channels_for_each_dec_layer[0]

        self.dec_fc = nn.Linear(num_latent_factors, self.num_dec_channels_first_layer * prod(self.feature_map_size))

        self.blocks = nn.ModuleList()
        self.blocks.append(SFCN_Like_Decoder_Block(num_in_channels=num_channels_for_each_dec_layer[0], num_out_channels=num_channels_for_each_dec_layer[1], kernel_size=3, deconv=False))
        for i in range(1, len(num_channels_for_each_dec_layer) - 2):
            self.blocks.append(SFCN_Like_Decoder_Block(num_in_channels=num_channels_for_each_dec_layer[i], num_out_channels=num_channels_for_each_dec_layer[i+1]))
        self.blocks.append(SFCN_Like_Decoder_Block(num_in_channels=num_channels_for_each_dec_layer[-2], num_out_channels=num_channels_for_each_dec_layer[-1], last_layer=True))

    def forward(self, x):
        x = self.dec_fc(x)
        x = x.view(-1, self.num_dec_channels_first_layer, *self.feature_map_size) # Unflattening
        for l in self.blocks:
            x = l(x)
        
        return x


class RajabSFCNVAERegularizedLM2(nn.Module):
        
    def __init__(self, input_shape, batch_size=1, dropout=0.0, risk_factors={}, affine_transforms={}, variatioanl_bottleneck_size=40, num_rf_ensembles=100):
        super(RajabSFCNVAERegularizedLM2, self).__init__()
        
        #self.last_layer_size = last_layer_size
        self.z_dim = variatioanl_bottleneck_size
        self.batch_size = batch_size
        self.risk_factors = risk_factors
        self.affine_transforms = affine_transforms
        self.num_rf_ensembles = num_rf_ensembles
        self.relu = nn.ReLU()

        num_channels=[1, 32, 64, 128, 256, 256, 64]
        num_channels_dec = num_channels.copy()
        num_channels_dec.reverse()
        self.enc_conv = ConvEncoder(num_channels_for_each_layer=num_channels)
        self.dropout3d = nn.Dropout3d(dropout)
        self.predictor_affine = PredictorAffine(num_channels[-1], [input_shape[0] // (2**(len(num_channels)-2)), input_shape[1] // (2**(len(num_channels)-2)), input_shape[2] // (2**(len(num_channels)-2))], affine_transforms)
        self.var_bottleneck = SFCN_Last_Block(num_channels[-1], [input_shape[0] // (2**(len(num_channels)-2)), input_shape[1] // (2**(len(num_channels)-2)), input_shape[2] // (2**(len(num_channels)-2))], 2*self.z_dim)
        self.var_dec = ConvDecoder([input_shape[0] // (2**(len(num_channels)-2)), input_shape[1] // (2**(len(num_channels)-2)), input_shape[2] // (2**(len(num_channels)-2))], self.z_dim + len(affine_transforms), num_channels_for_each_dec_layer=num_channels_dec)
        self.predictor_rfs = nn.ModuleList()
        for _ in range(self.num_rf_ensembles):
            self.predictor_rfs.append(PredictorRF(2*self.z_dim, risk_factors))
    
    def reparameterize(self, mu, std):
        epsilon = randn_like(std)
        z = mu + epsilon*std
        
        return z
    
    def forward(self, mri, affine_transforms):
        f = self.enc_conv(mri)
        f = self.dropout3d(f)
        estimated_affine_transforms = self.predictor_affine(f)
        f = self.var_bottleneck(f)
        estimated_risk_factors = []
        for i in range(self.num_rf_ensembles):
            estimated_risk_factors.append(self.predictor_rfs[i](f))

        z = self.reparameterize(f[:, :f.shape[-1]//2], f[:, f.shape[-1]//2:].mul(0.5).exp())
        # for rf in risk_factors:
        #     z = torch.cat([z, risk_factors[rf].unsqueeze(1)], 1)
        for tranform in affine_transforms:
            z = torch.cat([z, affine_transforms[tranform].unsqueeze(1)], 1)
        recon = self.var_dec(z)

        return estimated_risk_factors, estimated_affine_transforms, recon, z, f[:, :f.shape[-1]//2], f[:, f.shape[-1]//2:]
    
    def sample(self, n_samples, lower_cap=-2.0, higher_cap=2.0, device="cpu"):
        with torch.no_grad():

            mu = torch.randn((n_samples, self.z_dim + len(self.affine_transforms)), device=device)
            mu[mu < lower_cap] = lower_cap
            mu[mu > higher_cap] = higher_cap
            z = mu.view(n_samples, -1)
            
            samples = self.var_dec(z)

        return samples, z
