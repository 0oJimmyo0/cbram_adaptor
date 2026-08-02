import torch
import torch.nn as nn
import torch.nn.functional as F

from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
from models.generic_adapter import GenericBottleneckResidual
from models.interaction_adapter import CBraModInteractionAdapter


class CBraMod(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8, adapter_type="none", adapter_bottleneck=64,
                    adapter_heads=4, adapter_dropout=0.0, adapter_init_alpha=0.01,
                    adapter_gamma=1.0, adapter_zero_init_output=True,
                    adapter_seed=12345):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)
        self.native_axis_adapter = None
        self.generic_adapter = None
        self.adapter_type = "none"
        self._adapter_last_diagnostics = {}
        if str(adapter_type).strip().lower() != "none":
            self.enable_interaction_adapter(
                adapter_type=adapter_type,
                bottleneck=adapter_bottleneck,
                num_heads=adapter_heads,
                dropout=adapter_dropout,
                init_alpha=adapter_init_alpha,
                gamma=adapter_gamma,
                zero_init_output=adapter_zero_init_output,
                seed=adapter_seed,
            )

    def enable_interaction_adapter(
        self,
        adapter_type="channel_patch",
        bottleneck=64,
        num_heads=4,
        dropout=0.0,
        init_alpha=0.01,
        gamma=1.0,
        zero_init_output=True,
        seed=12345,
    ):
        """Attach the TMLR native interaction adapter after the encoder.

        This method is intentionally separate from checkpoint construction so
        the original CBraMod checkpoint can be loaded strictly before any new
        adapter parameters are introduced.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.native_axis_adapter = CBraModInteractionAdapter(
                d_model=self.encoder.layers[0].self_attn_s.embed_dim * 2,
                bottleneck=int(bottleneck),
                num_heads=int(num_heads),
                dropout=float(dropout),
                init_alpha=float(init_alpha),
                gamma=float(gamma),
                adapter_type=adapter_type,
                zero_init_output=bool(zero_init_output),
            )
        self.adapter_type = str(adapter_type).strip().lower()
        print(
            "[CBraMod adapter] interaction-aligned residual enabled: "
            f"type={self.adapter_type} bottleneck={bottleneck} heads={num_heads} "
            f"alpha={init_alpha} gamma={gamma} zero_init_output={bool(zero_init_output)}",
            flush=True,
        )
        return self.native_axis_adapter

    def enable_generic_adapter(
        self,
        bottleneck=64,
        init_alpha=0.01,
        gamma=1.0,
        zero_init_output=True,
        seed=12345,
        adapter_type="generic_bottleneck",
    ):
        """Attach a token-wise generic or parameter-matched axis-blind adapter."""
        if adapter_type not in {"generic_bottleneck", "axis_blind"}:
            raise ValueError(f"Unknown generic adapter type: {adapter_type!r}")
        if self.native_axis_adapter is not None or self.generic_adapter is not None:
            raise RuntimeError("Only one CBraMod adaptation family may be attached")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.generic_adapter = GenericBottleneckResidual(
                d_model=self.encoder.layers[0].self_attn_s.embed_dim * 2,
                bottleneck=int(bottleneck),
                init_alpha=float(init_alpha),
                gamma=float(gamma),
                zero_init_output=bool(zero_init_output),
            )
        self.adapter_type = str(adapter_type)
        return self.generic_adapter

    def get_adapter_diagnostics(self):
        if self.native_axis_adapter is not None:
            diagnostics = self.native_axis_adapter.get_diagnostics()
            diagnostics.update(self.native_axis_adapter.get_gradient_diagnostics())
            return diagnostics
        if self.generic_adapter is not None:
            diagnostics = self.generic_adapter.get_diagnostics()
            diagnostics.update(self.generic_adapter.get_gradient_diagnostics())
            return diagnostics
        diagnostics = {}
        return diagnostics

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        if self.native_axis_adapter is not None:
            feats = feats + self.native_axis_adapter(feats)
        if self.generic_adapter is not None:
            feats = feats + self.generic_adapter(feats)

        out = self.proj_out(feats)

        return out

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)
