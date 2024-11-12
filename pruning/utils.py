import torch
import numpy as np

from .dpf.mnn import MaskLinear


def get_importance(weight, imp_type):
    if imp_type == "L1":
        return weight.abs().mean(dim=1).detach().cpu().numpy()
    elif imp_type == "L2":
        return weight.pow(2).mean(dim=1).detach().cpu().numpy()
    else:
        raise ValueError("Invalid importance type. Choose 'L1' or 'L2'.")


def expand_mask(mask, in_features):
    return np.repeat(mask[:, np.newaxis], in_features, axis=1)


def expand_value_mask(dim_mask, in_features, num_heads):
    head_masks = np.tile(dim_mask, (num_heads, 1))  # [3, 64]
    flat_mask = head_masks.reshape(-1)  # [192]
    final_mask = np.tile(
        flat_mask[:, np.newaxis], (1, in_features)
    )  # [192, in_features]

    return final_mask


def get_threshold(importance_all, rate):
    return np.percentile(importance_all, rate * 100)


def get_mask(importance, threshold):
    return importance > threshold


def new_get_vit_masks(model, pruning_rate, imp_type, mag_type):
    importance_dict = {"qk": [], "v": [], "ffn1": []}
    q_importance = None
    k_importance = None
    all_importance = []

    if mag_type == "weight":
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight, imp_type)
                elif "value" in name:
                    # Get the weight importance
                    weight = module.weight
                    num_heads = weight.size(0) // 64
                    head_dim = 64

                    # Reshape weight to [num_heads, head_dim, in_features]
                    weight_reshaped = weight.view(num_heads, head_dim, -1)

                    if imp_type == "L1":
                        # Calculate importance per dimension position across all heads
                        dim_importance = (
                            weight_reshaped.abs()
                            .mean(dim=(0, 2))
                            .detach()
                            .cpu()
                            .numpy()
                        )  # [64]
                    elif imp_type == "L2":
                        dim_importance = (
                            weight_reshaped.pow(2)
                            .mean(dim=(0, 2))
                            .detach()
                            .cpu()
                            .numpy()
                        )  # [64]

                    # Store the dimension position importance directly
                    importance_dict["v"].append(dim_importance)  # [64]
                    all_importance.append(dim_importance)

                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict["qk"].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None

            elif (
                isinstance(module, MaskLinear) and "intermediate.dense" in name.lower()
            ):
                ffn_importance = get_importance(module.weight, imp_type)
                importance_dict["ffn1"].append(ffn_importance)
                all_importance.append(ffn_importance)

    elif mag_type == "grad":
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight.grad, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight.grad, imp_type)
                elif "value" in name:
                    grad = module.weight.grad
                    num_heads = grad.size(0) // 64
                    head_dim = 64

                    grad_reshaped = grad.view(num_heads, head_dim, -1)

                    if imp_type == "L1":
                        dim_importance = (
                            grad_reshaped.abs().mean(dim=(0, 2)).detach().cpu().numpy()
                        )
                    elif imp_type == "L2":
                        dim_importance = (
                            grad_reshaped.pow(2).mean(dim=(0, 2)).detach().cpu().numpy()
                        )

                    importance_dict["v"].append(dim_importance)
                    all_importance.append(dim_importance)

                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict["qk"].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None

            elif (
                isinstance(module, MaskLinear) and "intermediate.dense" in name.lower()
            ):
                ffn_importance = get_importance(module.weight.grad, imp_type)
                importance_dict["ffn1"].append(ffn_importance)
                all_importance.append(ffn_importance)

    # Calculate single threshold for all importances
    all_importance = np.concatenate(all_importance)
    threshold = get_threshold(all_importance, pruning_rate)

    # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
            if "query" in name:
                mask = importance_dict["qk"].pop(0) > threshold
                masks[f"{name}"] = expand_mask(mask, module.in_features)
            elif "value" in name:
                dim_importance = importance_dict["v"].pop(0)  # [64]
                dim_mask = dim_importance > threshold  # [64]
                num_heads = module.weight.size(0) // 64

                value_mask = expand_value_mask(dim_mask, module.in_features, num_heads)
                masks[f"{name}"] = value_mask

                output_mask = expand_value_mask(
                    dim_mask, module.out_features, num_heads
                ).T
                masks[name.replace("attention.value", "output.dense")] = output_mask

        elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
            mask = importance_dict["ffn1"].pop(0) > threshold
            masks[f"{name}"] = expand_mask(mask, module.in_features)

            # For FFN output, we use the transpose of the intermediate mask
            output_name = name.replace("intermediate.dense", "output.dense")
            output_module = next(
                m for n, m in model.named_modules() if n == output_name
            )
            masks[output_name] = expand_mask(mask, output_module.out_features).T

    return masks


def original_get_vit_masks(model, pruning_rate, imp_type, mag_type):
    importance_dict = {"qk": [], "v": [], "ffn1": []}
    q_importance = None
    k_importance = None
    all_importance = []

    if mag_type == "weight":
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight, imp_type)
                    importance_dict["v"].append(v_importance)
                    all_importance.append(v_importance)

                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict["qk"].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None

            elif (
                isinstance(module, MaskLinear) and "intermediate.dense" in name.lower()
            ):
                ffn_importance = get_importance(module.weight, imp_type)
                importance_dict["ffn1"].append(ffn_importance)
                all_importance.append(ffn_importance)

    elif mag_type == "grad":
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight.grad, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight.grad, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight.grad, imp_type)
                    importance_dict["v"].append(v_importance)
                    all_importance.append(v_importance)

                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict["qk"].append(qk_importance_avg)
                    all_importance.append(qk_importance_avg)
                    q_importance = None
                    k_importance = None

            elif (
                isinstance(module, MaskLinear) and "intermediate.dense" in name.lower()
            ):
                ffn_importance = get_importance(module.weight.grad, imp_type)
                importance_dict["ffn1"].append(ffn_importance)
                all_importance.append(ffn_importance)

    # Calculate single threshold for all importances
    all_importance = np.concatenate(all_importance)
    threshold = get_threshold(all_importance, pruning_rate)

    # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
            if "query" in name:
                mask = importance_dict["qk"].pop(0) > threshold
                masks[f"{name}"] = expand_mask(mask, module.in_features)
            elif "value" in name:
                mask = importance_dict["v"].pop(0) > threshold
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # For attention output, we use the transpose of the value mask
                masks[name.replace("attention.value", "output.dense")] = expand_mask(
                    mask, module.out_features
                ).T

        elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
            mask = importance_dict["ffn1"].pop(0) > threshold
            masks[f"{name}"] = expand_mask(mask, module.in_features)

            # For FFN output, we use the transpose of the intermediate mask
            output_name = name.replace("intermediate.dense", "output.dense")
            output_module = next(
                m for n, m in model.named_modules() if n == output_name
            )
            masks[output_name] = expand_mask(mask, output_module.out_features).T

    return masks


def get_vit_masks(model, pruning_rate, imp_type, mag_type):
    importance_dict = {"qk": [], "v": [], "ffn1": []}
    q_importance = None
    k_importance = None

    if mag_type == "weight":
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight, imp_type)
                    importance_dict["v"].append(v_importance)

                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict["qk"].append(qk_importance_avg)
                    q_importance = None
                    k_importance = None

            elif (
                isinstance(module, MaskLinear) and "intermediate.dense" in name.lower()
            ):
                ffn_importance = get_importance(module.weight, imp_type)
                importance_dict["ffn1"].append(ffn_importance)

    elif mag_type == "grad":
        for name, module in model.named_modules():
            if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
                if "query" in name:
                    q_importance = get_importance(module.weight.grad, imp_type)
                elif "key" in name:
                    k_importance = get_importance(module.weight.grad, imp_type)
                elif "value" in name:
                    v_importance = get_importance(module.weight.grad, imp_type)
                    importance_dict["v"].append(v_importance)

                if q_importance is not None and k_importance is not None:
                    qk_importance_avg = (q_importance + k_importance) / 2
                    importance_dict["qk"].append(qk_importance_avg)
                    q_importance = None
                    k_importance = None

            elif (
                isinstance(module, MaskLinear) and "intermediate.dense" in name.lower()
            ):
                ffn_importance = get_importance(module.weight.grad, imp_type)
                importance_dict["ffn1"].append(ffn_importance)

    # Calculate thresholds for each group
    thresholds = {}
    for key in importance_dict:
        if importance_dict[key]:
            importance_all = np.concatenate(importance_dict[key])
            thresholds[key] = np.percentile(importance_all, pruning_rate * 100)

    # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskLinear) and "attention.attention" in name.lower():
            if "query" in name:
                mask = importance_dict["qk"].pop(0) > thresholds["qk"]
                masks[f"{name}"] = expand_mask(mask, module.in_features)

            elif "value" in name:
                mask = importance_dict["v"].pop(0) > thresholds["v"]
                masks[f"{name}"] = expand_mask(mask, module.in_features)
                # For attention output, we use the transpose of the value mask
                masks[name.replace("attention.value", "output.dense")] = expand_mask(
                    mask, module.out_features
                ).T

        elif isinstance(module, MaskLinear) and "intermediate.dense" in name.lower():
            mask = importance_dict["ffn1"].pop(0) > thresholds["ffn1"]
            masks[f"{name}"] = expand_mask(mask, module.in_features)

            # For FFN output, we use the transpose of the intermediate mask
            output_name = name.replace("intermediate.dense", "output.dense")
            output_module = next(
                m for n, m in model.named_modules() if n == output_name
            )
            masks[output_name] = expand_mask(mask, output_module.out_features).T

    return masks


def apply_vit_masks(model, masks):
    for name, module in model.named_modules():
        with torch.no_grad():
            if isinstance(module, MaskLinear):
                if (
                    "attention.attention.query" in name
                    or "attention.attention.key" in name
                ):
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name:
                            mask_name = name.replace("key", "query")
                            param.data = (
                                torch.from_numpy(masks[mask_name]).float().cuda()
                            )

                elif "attention.attention.value" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name:
                            param.data = torch.from_numpy(masks[name]).float().cuda()

                elif "attention.output" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name:
                            param.data = torch.from_numpy(masks[name]).float().cuda()

                elif "intermediate.dense" in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name:
                            param.data = torch.from_numpy(masks[name]).float().cuda()

                elif "output.dense" in name and "attention" not in name:
                    for param_name, param in module.named_parameters():
                        if "mask" in param_name:
                            param.data = torch.from_numpy(masks[name]).float().cuda()


def vit_structured_pruning(model, pruning_rate, imp_type="L2"):
    masks = get_vit_masks(model, pruning_rate, imp_type)
    apply_vit_masks(model, masks)
    return model


def v_sparse_sum(model):
    state = model.state_dict()
    v_sparse_loss = 0.0
    for name, item in model.named_parameters():
        if "attention.attention.value.weight" in name.lower():
            m = name.replace("weight", "mask")

            v_sparse_loss += (
                -1 * torch.log(1 + (torch.abs(item) * state[m]))
            ).mean() + torch.log(torch.tensor([2.0])).cuda()

    return v_sparse_loss


def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():
        if "mask" in name and "fc" not in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        if "weight" in name and "MaskLinear" in name and "fc" not in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity


def cal_sparsity_nlp(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():
        if "embeddings" not in name and "mask" in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        if "embeddings" not in name and "weight" in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity
