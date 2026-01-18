"""
Dataset loader for CLIP fine-tuning.

Loads image-text pairs from CSV files for training/validation.
CSV format: image_path, text, label
- image_path: path to image file
- text: text description
- label: 1 for match, 0 for mismatch
"""

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageTextDataset(Dataset):
    """
    Dataset for image-text pairs with labels.
    
    Loads data from CSV file with columns: image_path, text, label
    """
    
    def __init__(
        self,
        csv_path: str,
        processor,
        image_base_path: Optional[str] = None,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with image_path, text, label columns
            processor: CLIP processor for preprocessing
            image_base_path: Base path to prepend to image paths (optional)
            transform: Additional image transformations (optional)
        """
        self.data = pd.read_csv(csv_path)
        self.processor = processor
        self.image_base_path = image_base_path or ""
        self.transform = transform
        
        # Validate CSV columns
        required_columns = ['image_path', 'text', 'label']
        missing_columns = set(required_columns) - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        
        # Validate labels
        invalid_labels = self.data[~self.data['label'].isin([0, 1])]
        if not invalid_labels.empty:
            logger.warning(f"Found {len(invalid_labels)} rows with invalid labels (not 0 or 1)")
            self.data = self.data[self.data['label'].isin([0, 1])]
        
        logger.info(f"Loaded dataset with {len(self.data)} samples from {csv_path}")
        logger.info(f"Label distribution: {self.data['label'].value_counts().to_dict()}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with processed inputs and label
        """
        row = self.data.iloc[idx]
        
        # Construct full image path
        image_path = os.path.join(self.image_base_path, row['image_path'])
        
        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            # Process inputs
            inputs = self.processor(
                text=[row['text']],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Remove batch dimension (will be added by DataLoader)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # Add label
            inputs['labels'] = torch.tensor(row['label'], dtype=torch.long)
            
            return inputs
            
        except FileNotFoundError:
            error_msg = (
                f"Image file not found for sample {idx}: {image_path}\n"
                f"Please verify:\n"
                f"  1. Image exists at the specified path\n"
                f"  2. --image_base_path is set correctly\n"
                f"  3. CSV contains correct relative paths"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = (
                f"Error loading sample {idx}:\n"
                f"  Image path: {image_path}\n"
                f"  Text: {row['text'][:50]}...\n"
                f"  Error: {str(e)}\n"
                f"Suggestion: Run with --validate_data to check dataset before training"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def create_dataloaders(
    train_csv: str,
    val_csv: Optional[str] = None,
    processor=None,
    image_base_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV (optional)
        processor: CLIP processor
        image_base_path: Base path for images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
        val_loader is None if val_csv is not provided
    """
    # Create datasets
    train_dataset = ImageTextDataset(
        train_csv,
        processor,
        image_base_path
    )
    
    val_dataset = None
    if val_csv:
        val_dataset = ImageTextDataset(
            val_csv,
            processor,
            image_base_path
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    logger.info(f"Created dataloaders - Train batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def validate_dataset(csv_path: str, image_base_path: str = "") -> dict:
    """
    Validate dataset CSV and check image accessibility.
    
    Args:
        csv_path: Path to CSV file
        image_base_path: Base path for images
        
    Returns:
        Dictionary with validation statistics
    """
    df = pd.read_csv(csv_path)
    
    stats = {
        'total_samples': len(df),
        'missing_images': 0,
        'invalid_labels': 0,
        'valid_samples': 0,
        'label_distribution': {}
    }
    
    # Check required columns
    required_columns = ['image_path', 'text', 'label']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return stats
    
    # Check each sample
    for idx, row in df.iterrows():
        image_path = os.path.join(image_base_path, row['image_path'])
        
        # Check if image exists
        if not os.path.exists(image_path):
            stats['missing_images'] += 1
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Check label
        if row['label'] not in [0, 1]:
            stats['invalid_labels'] += 1
            logger.warning(f"Invalid label {row['label']} at row {idx}")
            continue
        
        stats['valid_samples'] += 1
    
    # Get label distribution
    stats['label_distribution'] = df['label'].value_counts().to_dict()
    
    logger.info(f"Dataset validation complete: {stats}")
    
    return stats
