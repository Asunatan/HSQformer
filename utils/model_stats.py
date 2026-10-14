import logging

import torch
from torch import nn


def _get_profile_input_size(model: nn.Module) -> tuple[int, int]:
    for module in model.modules():
        patch_embed = getattr(module, "patch_embed", None)
        image_size = getattr(patch_embed, "img_size", None)
        if image_size is not None:
            if isinstance(image_size, int):
                return image_size, image_size
            return int(image_size[0]), int(image_size[1])
    return 224, 224


def log_model_statistics(model: nn.Module, logger: logging.Logger) -> None:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Total parameters: %.2fM", total_parameters / 1e6)
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError(
            "Model FLOPs profiling requires the 'thop' package; install it with 'pip install thop'."
        ) from error

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    input_tensor = torch.zeros(1, 3, 224, 224, device=device)
    training_states = [(module, module.training) for module in model.modules()]

    try:
        model.eval()
        with torch.no_grad():
            total_flops, total_params = profile(
                model, inputs=(input_tensor,), verbose=False
            )
        logger.info(
            "THOP-estimated FLOPs (input: 1x3x%dx%d): %.2fG",
            224,
            224,
            total_flops / 1e9,
        )
        logger.info("Total FLOPs: %.2fG", total_flops / 1e9)

        swintransformer = getattr(model, "swintransformer", None)
        convnext = getattr(model, "convnext", None)
        if isinstance(swintransformer, nn.Module) and isinstance(convnext, nn.Module):
            swin_flops, swin_params = profile(
                swintransformer, inputs=(input_tensor,), verbose=False
            )
            convnext_flops, convnext_params = profile(
                convnext, inputs=(input_tensor,), verbose=False
            )
            HSQformer_flops = total_flops - swin_flops - convnext_flops
            HSQformer_params = total_params - swin_params - convnext_params
            logger.info("HSQformer FLOPs: %.2fG", HSQformer_flops / 1e9)
            logger.info("HSQformer parameters: %.2fM", HSQformer_params / 1e6)

        else:
            logger.info(
                "THOP-estimated parameters: %.2fM", total_params / 1e6
            )

    except Exception as error:
        logger.warning("Could not profile model FLOPs: %s", error, exc_info=True)
    finally:
        for module, was_training in training_states:
            module.training = was_training
