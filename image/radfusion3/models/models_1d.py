import torch.nn as nn
from . import vision_backbones


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Adapted from:
        https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainall/2nd_level/seresnext101_192.py
    """

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        print("=" * 80)
        print("Using attention")
        print("=" * 80)

        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.weight = self.weight.type(torch.float32)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        self.b = self.b.type(torch.float32)

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim
        )

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1), self.weight


class RNNSequentialEncoder(nn.Module):
    """Model to encode series of encoded 2D CT slices using RNN

    Args:
        feature_size (int): number of features for input feature vector
        rnn_type (str): either lstm or gru
        hidden_size (int): number of hidden units
        bidirectional (bool): use bidirectional rnn
        num_layers (int): number of rnn layers
        dropout_prob (float): dropout probability
    """

    def __init__(
        self,
        feature_size: int,
        rnn_type: str = "lstm",
        hidden_size: int = 128,
        bidirectional: bool = True,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
    ):
        super(RNNSequentialEncoder, self).__init__()

        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers

        if self.rnn_type not in ["LSTM", "GRU"]:
            raise Exception("RNN type has to be either LSTM or GRU")

        self.rnn = getattr(nn, rnn_type)(
            self.feature_size,
            self.hidden_size,
            batch_first=False,  # input is (Slice, Batch, Feature) after transpose
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            bidirectional=bidirectional,
        )

    def forward(self, x, lengths=None):
        # Without `lengths`, the RNN processes every one of the num_slices
        # timesteps as-is, including the zero-padded ones appended by
        # fix_series_slice_number() for scans shorter than num_slices. For a
        # bidirectional RNN this is worse than it sounds: the backward
        # direction starts at the last timestep and runs toward the real
        # data, so for any short scan it spends its first steps consuming
        # pure zero-padding before reaching genuine slices — contaminating
        # the hidden state at every real position with a padding-induced
        # offset before pooling ever sees it. Pooling-level masking (in
        # aggregate()) only hides padded positions from the final reduction;
        # it can't undo this upstream corruption of the "real" positions'
        # own hidden states.
        #
        # pack_padded_sequence/pad_packed_sequence make cuDNN skip padded
        # timesteps entirely per-sample, so the backward pass genuinely
        # starts at each sample's real last slice instead of at the
        # fixed padded end. No parameters or output shape change.
        x = x.transpose(0, 1)  # (Slice, Batch, Feature)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=False, enforce_sorted=False
            )
            packed, _ = self.rnn(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed, batch_first=False, total_length=x.size(0)
            )
        else:
            x, _ = self.rnn(x)  # (Slice, Batch, Feature)
        x = x.transpose(0, 1)  # (Batch, Slice, Feature)
        return x


def get_transformer(n_layers, seq_input_size, nhead, hidden_size, dropout_prob):
    layers = torch.nn.Sequential()

    for _ in range(n_layers):
        layers.append(
            nn.TransformerEncoderLayer(
                d_model=seq_input_size,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=hidden_size,
                dropout=dropout_prob,
            )
        )

    return layers


class Model1D(nn.Module):
    def __init__(self, cfg, num_classes=1):
        super(Model1D, self).__init__()

        # rnn input size — read directly from config, no need to load the backbone
        seq_input_size = cfg.dataset.feature_size
        if cfg.trainer.position_encoding:
            seq_input_size += 1

        if cfg.dataset.contextualize_slice:
            seq_input_size = seq_input_size * 3

        # classifier input size
        cls_input_size = cfg.model.seq_encoder.hidden_size

        if cfg.model.seq_encoder.rnn_type == "transformer":
            cls_input_size = seq_input_size
            self.seq_encoder = get_transformer(
                n_layers=cfg.model.seq_encoder.num_layers,
                seq_input_size=seq_input_size,
                nhead=16,
                hidden_size=cfg.model.seq_encoder.hidden_size,
                dropout_prob=cfg.model.seq_encoder.dropout_prob,
            )
        elif cfg.model.seq_encoder.rnn_type in ["LSTM", "GRU"]:
            if cfg.model.seq_encoder.bidirectional:
                cls_input_size = cls_input_size * 2
            self.seq_encoder = RNNSequentialEncoder(
                seq_input_size, **cfg.model.seq_encoder
            )
        else:
            raise Exception("")

        # RNNSequentialEncoder accepts a `lengths` kwarg (to skip padded
        # timesteps via pack_padded_sequence); the transformer branch above
        # doesn't, so only pass lengths through when we actually built an RNN.
        self._is_rnn_encoder = cfg.model.seq_encoder.rnn_type in ["LSTM", "GRU"]

        if "attention" in cfg.model.aggregation:
            self.attention = Attention(cls_input_size, cfg.dataset.num_slices)

        if cfg.model.aggregation == "attention+max":
            cls_input_size = cls_input_size * 2

        # Normalize only the raw CNN feature block, not the position-encoding
        # column appended after it (dataset_base.py: `arr = np.concatenate(
        # [arr, pos[:, None]], axis=1)`, position is the LAST column). Position
        # is already rescaled to [0, 1] (see dataset_base.py's fix for LSTM
        # gradient explosion from unnormalized slice positions). Folding it
        # into a single joint LayerNorm over all `seq_input_size` dims would
        # renormalize it using the CNN features' per-token mean/std instead of
        # its own consistent range: since position is 1 value out of ~2049, it
        # barely affects that mean/std, so it would get rescaled by whatever
        # the CNN activation statistics happen to be for that slice — noise
        # that varies per token/sample and has nothing to do with slice order.
        # That destroys the monotonic 0->1 position signal the RNN needs.
        # Splitting the normalization keeps the CNN features from saturating
        # the RNN gates while leaving position untouched.
        #
        # This assumes contextualize_slice=False (the default, and what every
        # run_classify_*.sh currently uses). When contextualize_slice=True,
        # dataset_1d.py's contextualize_slice() triples/interleaves the
        # position column into slice-difference blocks, so "last column =
        # position" no longer holds; that path isn't exercised by any current
        # script, so it's intentionally left using the old joint-normalization
        # behavior below rather than guessing at the right split.
        self._norm_features_only = (
            cfg.trainer.position_encoding and not cfg.dataset.contextualize_slice
        )
        norm_size = (
            cfg.dataset.feature_size if self._norm_features_only else seq_input_size
        )
        self.input_norm = nn.LayerNorm(norm_size)
        # self.batch_norm_layer = torch.nn.BatchNorm1d(cls_input_size)
        self.classifier = nn.Linear(cls_input_size, num_classes)
        self.cfg = cfg

    def forward(self, x, get_features=False, mask=None):
        if self._norm_features_only:
            feats, pos = x[..., :-1], x[..., -1:]
            x = torch.cat([self.input_norm(feats), pos], dim=-1)
        else:
            x = self.input_norm(x)
        if self._is_rnn_encoder and mask is not None:
            # mask: (Batch, Slice), 1=real, 0=padded. Real slice count per
            # sample, clamped to >=1 since pack_padded_sequence rejects
            # zero-length sequences (shouldn't occur — every sample has at
            # least one real slice — but clamp defensively).
            lengths = mask.sum(dim=1).clamp(min=1).long()
            x = self.seq_encoder(x, lengths=lengths)
        else:
            x = self.seq_encoder(x)
        x, w = self.aggregate(x, mask)
        # x = self.batch_norm_layer(x)
        pred = self.classifier(x)
        return pred, x

    def aggregate(self, x, mask=None):
        if self.cfg.model.aggregation == "attention":
            return self.attention(x, mask)
        elif self.cfg.model.aggregation == "attention+max":
            max_pool, _ = torch.max(x, 1)
            attn_pool, w = self.attention(x, mask)
            x = torch.cat((max_pool, attn_pool), 1)
            return x, w
        elif self.cfg.model.aggregation == "mean":
            x = torch.mean(x, 1)
            return x, None
        elif self.cfg.model.aggregation == "max":
            if mask is not None:
                # mask: (Batch, Slice), 1=real, 0=padded
                # set padded positions to -inf so they never win the max
                inf_mask = (1 - mask).bool().unsqueeze(-1).expand_as(x)
                x = x.masked_fill(inf_mask, float('-inf'))
            x, _ = torch.max(x, 1)
            return x, None
        else:
            raise Exception(
                "Aggregation method should be one of 'attention', 'mean' or 'max'"
            )
