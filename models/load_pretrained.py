"""Load released Smart Turn weights into the PyTorch ``SmartTurnV3Model``.

pipecat-ai/smart-turn-v3 ships ONNX files only. The fp32 ("gpu") export keeps most parameter
names (``inner.<torch name>``) but the Linear weights inside the encoder and the attention
pooling were exported as anonymous MatMul initializers (``val_17`` ...), stored transposed.
We name each one by following its MatMul output to the ``Add`` that consumes a named bias.
Whisper's k_proj has no bias, so it is the anonymous MatMul that comes straight after a q_proj.

The result is loaded with ``strict=True`` and checked against onnxruntime, so a mapping error
cannot pass silently. This is a weight copy, not a graph conversion (no onnx2torch needed).
"""
from __future__ import annotations

import re

import numpy as np
import torch

from .smart_turn import SmartTurnV3Model, build_empty

_INNER = "inner."


def onnx_to_state_dict(onnx_path: str) -> dict[str, torch.Tensor]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    state: dict[str, torch.Tensor] = {}
    for name, arr in inits.items():
        if name.startswith(_INNER):
            state[name[len(_INNER):]] = torch.from_numpy(arr.copy())

    last_resolved = None
    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        w_name = node.input[1]
        if w_name not in inits or w_name.startswith(_INNER):
            continue
        prefix = None
        for nxt in consumers.get(node.output[0], []):
            if nxt.op_type == "Add":
                biases = [i for i in nxt.input if i.startswith(_INNER) and i.endswith(".bias")]
                if len(biases) == 1:
                    prefix = biases[0][len(_INNER):-len(".bias")]
                    break
        if prefix is None:
            if last_resolved and last_resolved.endswith("self_attn.q_proj"):
                prefix = re.sub(r"q_proj$", "k_proj", last_resolved)
            else:
                raise ValueError(f"Cannot name ONNX weight {w_name} (after {last_resolved})")
        key = prefix + ".weight"
        if key in state:
            raise ValueError(f"Duplicate mapping for {key}")
        state[key] = torch.from_numpy(inits[w_name].T.copy())  # MatMul stores [in, out]
        last_resolved = prefix

    return state


def load_from_onnx(onnx_path: str, whisper_name: str = "openai/whisper-tiny",
                   whisper_revision: str | None = None) -> SmartTurnV3Model:
    model = build_empty(whisper_name, whisper_revision)
    state = onnx_to_state_dict(onnx_path)
    model.load_state_dict(state, strict=True)
    return model.eval()


def download_onnx(cfg: dict, key: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=cfg["hf"]["smart_turn_model_repo"],
        filename=cfg["hf"]["smart_turn_onnx"][key],
        revision=cfg["hf"]["smart_turn_model_revision"],
    )


def load_smart_turn_v32(cfg: dict) -> tuple[SmartTurnV3Model, str]:
    """Stock Smart Turn v3.2 (fp32) as a trainable PyTorch model. Returns (model, onnx_path)."""
    path = download_onnx(cfg, "v3.2_gpu")
    return load_from_onnx(path, cfg["hf"]["whisper"]["tiny"]), path


@torch.no_grad()
def max_diff_vs_onnx(model: SmartTurnV3Model, onnx_path: str, features: np.ndarray) -> float:
    """Largest |p_torch - p_onnx| over a batch of (B, 80, 800) features."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    p_onnx = sess.run(None, {"input_features": features.astype(np.float32)})[0]
    model = model.eval().cpu().float()
    p_torch = model(torch.from_numpy(features.astype(np.float32)))["logits"].numpy()
    return float(np.abs(p_onnx.reshape(-1) - p_torch.reshape(-1)).max())
