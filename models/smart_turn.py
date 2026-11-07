"""Smart Turn v3 architecture, copied unchanged from pipecat-ai/smart-turn train.py.

Source: https://github.com/pipecat-ai/smart-turn (commit 4786657e, BSD 2-Clause, (c) Daily).
Do not change the layers, init or forward pass: our numbers have to be comparable with theirs.
The only addition is ``build_from_whisper`` (the same call train.py makes) and a ``pos_weight``
argument so the training loop can pass a fixed class weight instead of the per-batch one.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import softmax
from transformers import WhisperConfig, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder


class SmartTurnV3Model(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        config.max_source_positions = 400
        self.encoder = WhisperEncoder(config)

        hidden_size = config.d_model

        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward_logits(self, input_features: torch.Tensor) -> torch.Tensor:
        """Raw (pre-sigmoid) logits, shape (B, 1). Used by our training loop."""
        hidden_states = self.encoder(input_features=input_features).last_hidden_state
        attention_weights = softmax(self.pool_attention(hidden_states), dim=1)
        pooled = torch.sum(hidden_states * attention_weights, dim=1)
        return self.classifier(pooled)

    def forward(self, input_features, labels=None, pos_weight=None):
        """Same contract as upstream: returns {"loss", "logits"} where "logits" are sigmoid probs."""
        logits = self.forward_logits(input_features)

        if labels is not None:
            if pos_weight is None:
                # Upstream behaviour: weight from batch statistics.
                pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
            return {"loss": loss, "logits": torch.sigmoid(logits.detach())}

        return {"logits": torch.sigmoid(logits)}


def build_from_whisper(whisper_name: str, revision: str | None = None) -> SmartTurnV3Model:
    """Fresh Smart Turn model with encoder weights from a Whisper checkpoint (as upstream train.py)."""
    return SmartTurnV3Model.from_pretrained(
        whisper_name, revision=revision, num_labels=1, ignore_mismatched_sizes=True
    )


def build_empty(whisper_name: str = "openai/whisper-tiny", revision: str | None = None) -> SmartTurnV3Model:
    """Architecture only (random weights); used before loading converted ONNX weights."""
    config = WhisperConfig.from_pretrained(whisper_name, revision=revision)
    return SmartTurnV3Model(config)


def encoder_layer_groups(model: SmartTurnV3Model) -> dict[str, list[nn.Parameter]]:
    """Parameter groups for the two learning rates (encoder vs head)."""
    enc, head = [], []
    for name, p in model.named_parameters():
        (enc if name.startswith("encoder.") else head).append(p)
    return {"encoder": enc, "head": head}


def freeze_lower_encoder_layers(model: SmartTurnV3Model, n_frozen: int) -> None:
    """Freeze conv stem + the first ``n_frozen`` transformer layers (experiment E6)."""
    enc = model.encoder
    for m in [enc.conv1, enc.conv2]:
        for p in m.parameters():
            p.requires_grad = False
    for layer in enc.layers[:n_frozen]:
        for p in layer.parameters():
            p.requires_grad = False
