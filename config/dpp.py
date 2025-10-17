# utils/ddp_utils.py
import os
import torch
import torch.distributed as dist

def setup_distributed_training():
    """Initializes the distributed process group."""
    if 'RANK' not in os.environ:
        # Not in a distributed environment
        return False

    # Assumes torchrun or a similar launcher is used
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    print(f"[Rank {rank}] Distributed process started on cuda:{local_rank}.")
    return True

def cleanup_distributed_training():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    """Checks if the current process is the main one (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0