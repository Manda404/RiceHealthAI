"""
Custom PyTorch Dataset for RiceHealthAI.

Loads images and labels from a DataFrame, applies transformations,
encodes labels consistently across train/valid/test,
and automatically detects and uses the best device (CPU/GPU/MPS).
"""

from typing import Callable, Optional, Tuple, List
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from ricehealthai.infrastructure.logger import setup_logger
from ricehealthai.core.utils import get_device

logger = setup_logger(__name__)


class RiceLeafDataset(Dataset):
    """
    Custom Dataset for Rice Leaf Disease classification.

    Each item returns:
        - an image tensor (after transformations)
        - its encoded label (int)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        label_encoder: Optional[LabelEncoder] = None,
        device: Optional[str] = None,
        fit_encoder: bool = False,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Must contain 'image_path' and 'label' columns.
        transform : Callable, optional
            Transformations applied to each image (e.g., Resize, Normalize, Augment).
        label_encoder : LabelEncoder, optional
            Existing LabelEncoder instance from training dataset.
        device : str, optional
            Device to which the tensors will be moved ("cpu", "cuda", or "mps").
            If None, automatically detects the best available device.
        fit_encoder : bool, default=False
            If True, fits the LabelEncoder on this dataset (used for training set only).
        """
        if not {"image_path", "label"}.issubset(dataframe.columns):
            raise ValueError("DataFrame must contain 'image_path' and 'label' columns.")

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.device = torch.device(device) if device else get_device(verbose=False)

        # Handle LabelEncoder
        if label_encoder is None and not fit_encoder:
            raise ValueError(
                "LabelEncoder must be provided unless fit_encoder=True (for training set)."
            )

        if fit_encoder:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.dataframe["label"])
            logger.info(f"Fitted LabelEncoder with classes: {self.label_encoder.classes_}")
        else:
            self.label_encoder = label_encoder

        # Encode labels using the shared encoder
        self.encoded_labels = self.label_encoder.transform(self.dataframe["label"])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.dataframe.loc[idx, "image_path"]
        label = self.encoded_labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # fallback transform
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Move tensors to device
        image = image.to(self.device)
        label = torch.tensor(label, dtype=torch.long, device=self.device)

        return image, label

    def get_label_encoder(self) -> LabelEncoder:
        """Return the fitted LabelEncoder."""
        return self.label_encoder

    def get_class_names(self) -> List[str]:
        """Return the class names in label order."""
        return list(self.label_encoder.classes_)
