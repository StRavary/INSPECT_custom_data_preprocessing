import torch.nn as nn
from . import vision_backbones


class Model2D(nn.Module):
    def __init__(self, cfg, num_class=1, **kwargs):
        super(Model2D, self).__init__()

        # define cnn model
        model_function = getattr(vision_backbones, cfg.model.model_name)
        model_kwargs = {}
        if hasattr(cfg.model, 'checkpoint_path'):
            model_kwargs['checkpoint_path'] = cfg.model.checkpoint_path
        self.model, self.feature_dim = model_function(**model_kwargs)

        # Unlike Model1D, this classifier previously had zero regularization
        # between the pooled backbone features and the linear head — no
        # dropout option even existed. For a full end-to-end fine-tune with
        # weighted_sample=true repeatedly oversampling the same small
        # positive pool, that's an open door to fast memorization. Opt-in via
        # cfg.model.dropout_prob (absent/0 on every existing model config, so
        # this is a no-op everywhere except where explicitly set, e.g.
        # resnetv2_ct_pretrain.yaml).
        dropout_prob = getattr(cfg.model, "dropout_prob", 0.0)
        self.classifier_dropout = (
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        )

        self.classifier = nn.Linear(self.feature_dim, num_class)
        self.cfg = cfg
        self.get_features = cfg.get_features

    def freeze_backbone(self):
        """Disable grad on the backbone so only the classifier head trains.

        Used for a linear-probe warmup phase (cfg.trainer.freeze_backbone_epochs)
        before unfreezing for full end-to-end fine-tuning -- see
        ClassificationLightningModel.on_train_epoch_start.
        """
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x, mask=None, get_features=False):
        x = self.model(x)
        pred = self.classifier(self.classifier_dropout(x))
        if get_features:
            # x is the raw (pre-dropout) pooled backbone feature — this is
            # what run_featurize.py saves as CNN embeddings for the
            # downstream RNN stage, so it must stay unaffected by dropout
            # regardless of train/eval mode (dropout only perturbs the
            # classifier's own input, never the returned features).
            return pred, x
        else:
            return pred
