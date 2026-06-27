import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def is_distributed_available():
    return dist.is_available() and dist.is_initialized()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        device=device,
    )


def cleanup_distributed():
    if is_distributed_available():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def wrap_model_for_distributed(model: nn.Module, context: DistributedContext):
    if not context.enabled:
        return model

    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )

    return DistributedDataParallel(model)


def load_checkpoint_state(model: nn.Module, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    target_model = unwrap_model(model)
    try:
        target_model.load_state_dict(checkpoint)
    except RuntimeError:
        normalized_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            normalized_checkpoint[key] = value
        target_model.load_state_dict(normalized_checkpoint)


def rank_zero_print(context: DistributedContext, *args, **kwargs):
    if context.is_main:
        print(*args, **kwargs)
