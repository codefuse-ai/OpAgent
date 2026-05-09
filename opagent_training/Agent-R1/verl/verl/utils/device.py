# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()

# Workaround: When running in GPU containers, the CUDA driver version check may
# fail (e.g., old host driver visible from inside the container) even though GPUs
# are physically present and usable via the container runtime. In that case,
# torch.cuda.is_available() returns False, which causes init_process_group to
# build an invalid backend string "cpu:gloo,cpu:nccl". Fall back to checking if
# NVIDIA device nodes exist or if CUDA_VISIBLE_DEVICES is set non-empty.
if not is_cuda_available:
    import os as _os
    _cuda_vis = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _has_nvidia_dev = any(_os.path.exists(f"/dev/nvidia{i}") for i in range(16))
    _force = _os.environ.get("VERL_FORCE_CUDA", "")
    if _force == "1" or (_has_nvidia_dev and _cuda_vis != ""):
        logger.warning(
            "torch.cuda.is_available() returned False but GPU devices appear present "
            f"(CUDA_VISIBLE_DEVICES={_cuda_vis!r}, /dev/nvidia* exists={_has_nvidia_dev}). "
            "Forcing is_cuda_available=True. Set VERL_FORCE_CUDA=0 to disable this workaround."
        )
        is_cuda_available = True


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    if is_npu_available:
        return "hccl"
    else:
        # default to nccl
        return "nccl"


def set_expandable_segments(enable: bool) -> None:
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
