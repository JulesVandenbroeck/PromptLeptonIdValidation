import torch
import torch.nn as nn
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

import torch.nn.functional as F


class CustomParticleTransformer(nn.Module):
    def __init__(self, **kwargs):

        pf_dim = kwargs.pop('pf_input_dim')
        sv_dim = kwargs.pop('sv_input_dim')
        self.hlf_dim = kwargs.pop('highlevel_dim', 0)
        num_classes = kwargs.get('num_classes')
        common_dim = kwargs.get('input_dim', 128)

        super().__init__()

        self.pf_embed = nn.Conv1d(pf_dim, common_dim, 1)
        self.sv_embed = nn.Conv1d(sv_dim, common_dim, 1)

        self.pf_type_emb = nn.Parameter(torch.randn(1, common_dim, 1) * 0.02)
        self.sv_type_emb = nn.Parameter(torch.randn(1, common_dim, 1) * 0.02)

        self.mod = ParticleTransformer(**kwargs)
        self.mod.fc = nn.Identity()

        self.part_norm = nn.LayerNorm(common_dim)

        self.hlf_embed = nn.Sequential(
            nn.Linear(self.hlf_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(common_dim + 64, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, pf_features, pf_points, pf_vectors, pf_mask,
                sv_features, sv_points, sv_vectors, sv_mask, high_level):

        pf_combined = torch.cat([pf_points, pf_features], dim=1)
        sv_combined = torch.cat([sv_points, sv_features], dim=1)

        pf_x = (self.pf_embed(pf_combined) + self.pf_type_emb) * pf_mask
        sv_x = (self.sv_embed(sv_combined) + self.sv_type_emb) * sv_mask

        x = torch.cat([pf_x, sv_x], dim=2)
        v = torch.cat([pf_vectors, sv_vectors], dim=2)
        mask = torch.cat([pf_mask, sv_mask], dim=2).bool()

        jet_x = self.mod(x, v=v, mask=mask)
        jet_x = self.part_norm(jet_x)

        hlf_x = self.hlf_embed(high_level.view(high_level.size(0), -1))
        combined = torch.cat([jet_x, hlf_x], dim=1)

        logits = self.cls_head(combined)
        return torch.nan_to_num(logits, nan=0.0)


def get_model(data_config, **kwargs):

    pf_total_dim = len(
        data_config.input_dicts['pf_features']+data_config.input_dicts['pf_points'])
    sv_total_dim = len(
        data_config.input_dicts['sv_features']+data_config.input_dicts['sv_points'])
    num_classes = len(data_config.label_value)
    hlf_dim = len(data_config.input_dicts['high_level'])

    cfg = dict(
        num_classes=num_classes,
    )

    cfg.update(
        pf_input_dim=pf_total_dim,
        sv_input_dim=sv_total_dim,
        num_classes=num_classes,
        highlevel_dim=hlf_dim,
        input_dim=128,
        pair_input_dim=4,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        activation='gelu',
        trim=True,
    )

    # dont' use this!!! - why? no idea...
    # cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = CustomParticleTransformer(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info
