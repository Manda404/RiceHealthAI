"""
Utility functions for RiceHealthAI.

Includes device management, tensor transfer utilities,
and simple console color helpers.
"""

import torch, random, numpy as np
from ricehealthai.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class clr:
    """Simple ANSI color helper for clean console output."""

    R = "\033[91m"   # Rouge
    G = "\033[92m"   # Vert
    Y = "\033[93m"   # Jaune
    B = "\033[94m"   # Bleu
    M = "\033[95m"   # Magenta
    C = "\033[96m"   # Cyan
    E = "\033[0m"    # Reset (fin de couleur)

    # Alias de compatibilité avec ton code précédent
    S = R             # S = Start color → rouge maintenant

def get_device(verbose: bool = True) -> torch.device:
    """
    Automatically select the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        if verbose:
            logger.info(f"Using CUDA GPU: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        if verbose:
            logger.info("Using CPU (no GPU available)")
    return device

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preview_dataloader(name: str, loader, n_batches: int = 3):
    print("\n" + "=" * 60)
    print(f"{name} DataLoader Preview")
    print("=" * 60 + "\n")
    for k, data in enumerate(loader):
        image, targets = data
        print(
            clr.S + f"Batch: {k}" + clr.E, "\n"
            + clr.S + "Image:" + clr.E, image.shape, "\n"
            + clr.S + "Targets:" + clr.E, targets, "\n"
            + "=" * 50
        )
        if k == n_batches:
            break