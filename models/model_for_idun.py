import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbramod import CBraMod  # Ensure CBraMod handles single channel input properly

class Model(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
            self.backbone.proj_out = nn.Identity()

        # The head projects flattened backbone features (of size 1*30*200 = 6000) to 512.
        self.head = nn.Sequential(
            nn.Linear(1*30*200, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Transformer encoder for sequence-level encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048,
            batch_first=True, activation=F.gelu, norm_first=True)
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(512, param.num_of_classes)

    def forward(self, x):
        bz, ch_num, seq_len, epoch_size = x.shape
        x = x.contiguous().view(bz, ch_num, seq_len, epoch_size)
        epoch_features = self.backbone(x)
        epoch_features = epoch_features.contiguous().view(bz, -1)  # => [64, 6000]
        epoch_features = self.head(epoch_features) # [64, 512]
        seq_features = self.sequence_encoder(epoch_features)
        out = self.classifier(seq_features) # [64, 5]
        
        return out