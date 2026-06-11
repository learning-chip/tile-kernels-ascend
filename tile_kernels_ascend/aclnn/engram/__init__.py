import torch

def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "ACLNN backend: no torch_npu APIs exist for engram_hash. "
        "This is a custom n-gram hashing architecture with no Ascend NPU kernel support."
    )


def engram_gate_fwd(
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
):
    raise NotImplementedError(
        "ACLNN backend: no torch_npu APIs exist for engram_gate_fwd. "
        "This is a custom gated attention mechanism with no Ascend NPU kernel support."
    )


def engram_gate_bwd(
    grad_output: torch.Tensor,
    saved_tensors: tuple,
) -> tuple:
    raise NotImplementedError(
        "ACLNN backend: no torch_npu APIs exist for engram_gate_bwd. "
        "Backward pass for engram_gate_fwd is not available on Ascend NPU."
    )


def grad_w_reduce(
    grad_w_partial: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "ACLNN backend: no torch_npu APIs exist for grad_w_reduce. "
        "This is a custom gradient reduction operation specific to engram architecture."
    )


def fused_weight(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "ACLNN backend: no torch_npu APIs exist for fused_weight. "
        "This is a custom weight fusion operation for engram architecture with no NPU kernel."
    )
