"""Optimizer parameter grouping shared by training and lightweight tests."""


def build_lora_optimizer_params(
    learnable_named_params,
    *,
    model_family,
    use_mem,
):
    """Protect gradient-masked Qwen2.5 embedding rows from AdamW decay."""
    if model_family != "qwen2_5_vl" or not use_mem:
        return list(learnable_named_params.values())

    memory_embedding_params = [
        parameter
        for name, parameter in learnable_named_params.items()
        if "embed_tokens" in name
    ]
    if len(memory_embedding_params) != 1:
        raise ValueError(
            "Qwen2.5 Full MemVLN expects exactly one trainable "
            "embed_tokens parameter, found "
            f"{len(memory_embedding_params)}."
        )
    memory_embedding_param_ids = {
        id(parameter) for parameter in memory_embedding_params
    }
    other_params = [
        parameter
        for parameter in learnable_named_params.values()
        if id(parameter) not in memory_embedding_param_ids
    ]
    groups = []
    if other_params:
        groups.append({"params": other_params})
    groups.append(
        {
            "params": memory_embedding_params,
            # The gradient hook masks every non-Memory row. AdamW decay must
            # also be disabled or those rows would change with zero gradients.
            "weight_decay": 0.0,
        }
    )
    return groups


__all__ = ["build_lora_optimizer_params"]
