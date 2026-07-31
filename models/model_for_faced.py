import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            checkpoint = torch.load(param.foundation_dir, map_location=map_location)
            load_report = self.backbone.load_state_dict(checkpoint, strict=True)
            print(
                "[CBraMod checkpoint] strict load passed: "
                f"missing={len(load_report.missing_keys)} unexpected={len(load_report.unexpected_keys)}",
                flush=True,
            )

        adapter_type = str(getattr(param, 'adapter_type', 'none')).strip().lower()
        if adapter_type != 'none':
            self.backbone.enable_interaction_adapter(
                adapter_type=adapter_type,
                bottleneck=int(getattr(param, 'adapter_bottleneck', 64)),
                num_heads=int(getattr(param, 'adapter_heads', 4)),
                dropout=float(getattr(param, 'adapter_dropout', 0.0)),
                init_alpha=float(getattr(param, 'adapter_init_alpha', 0.01)),
                gamma=float(getattr(param, 'adapter_gamma', 1.0)),
                zero_init_output=bool(getattr(param, 'adapter_zero_init_output', True)),
                seed=int(getattr(param, 'adapter_seed', 12345)),
            )
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 10 * 200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 10 * 200, 10 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


