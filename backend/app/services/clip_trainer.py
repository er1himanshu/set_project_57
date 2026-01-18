"""
CLIP model fine-tuning utilities for image-text match/mismatch training.

This module provides functionality to fine-tune CLIP models on custom datasets
with image-text pairs labeled as match or mismatch.
"""

import os
import torch
import pandas as pd
import logging
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AdamW
from tqdm import tqdm

from ..config import (
    CLIP_MODEL_NAME,
    CLIP_TRAIN_BATCH_SIZE,
    CLIP_TRAIN_LEARNING_RATE,
    CLIP_TRAIN_EPOCHS,
    CLIP_CHECKPOINT_DIR
)

logger = logging.getLogger(__name__)


class CLIPMatchDataset(Dataset):
    """
    Dataset for CLIP fine-tuning with image-text match/mismatch pairs.
    
    Expected CSV format:
        image_path,text,label
        /path/to/image1.jpg,"red leather handbag",1
        /path/to/image2.jpg,"blue cotton t-shirt",0
    
    where label: 1 = match, 0 = mismatch
    """
    
    def __init__(
        self,
        csv_path: str,
        processor: CLIPProcessor,
        image_base_path: Optional[str] = None
    ):
        """
        Initialize dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file with columns: image_path, text, label
            processor: CLIP processor for preprocessing
            image_base_path: Optional base path to prepend to image_path entries
        """
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.image_base_path = image_base_path or ""
        
        # Validate dataset
        self._validate_dataset()
        
        logger.info(f"Loaded dataset with {len(self.df)} samples")
        logger.info(f"Match samples: {(self.df['label'] == 1).sum()}")
        logger.info(f"Mismatch samples: {(self.df['label'] == 0).sum()}")
    
    def _validate_dataset(self):
        """Validate dataset has required columns and valid data."""
        required_cols = ['image_path', 'text', 'label']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        if self.df[required_cols].isnull().any().any():
            raise ValueError("Dataset contains null values")
        
        # Validate labels are 0 or 1
        if not self.df['label'].isin([0, 1]).all():
            raise ValueError("Labels must be 0 (mismatch) or 1 (match)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from dataset.
        
        Returns:
            dict: Preprocessed inputs with 'pixel_values', 'input_ids', 'attention_mask', 'label'
        """
        row = self.df.iloc[idx]
        
        # Construct full image path
        image_path = os.path.join(self.image_base_path, row['image_path'])
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
        
        # Process image and text
        inputs = self.processor(
            text=row['text'],
            images=image,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=77
        )
        
        # Remove batch dimension and add label
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.float32)
        }


class CLIPTrainer:
    """
    Trainer for fine-tuning CLIP models on match/mismatch datasets.
    """
    
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        learning_rate: float = CLIP_TRAIN_LEARNING_RATE,
        device: Optional[str] = None
    ):
        """
        Initialize CLIP trainer.
        
        Args:
            model_name: Name or path of base CLIP model
            learning_rate: Learning rate for training
            device: Device to train on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        
        logger.info(f"Initializing CLIP trainer on device: {self.device}")
        logger.info(f"Loading base model: {model_name}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        logger.info("CLIP trainer initialized successfully")
    
    def train(
        self,
        train_csv: str,
        val_csv: Optional[str] = None,
        epochs: int = CLIP_TRAIN_EPOCHS,
        batch_size: int = CLIP_TRAIN_BATCH_SIZE,
        output_dir: str = CLIP_CHECKPOINT_DIR,
        image_base_path: Optional[str] = None,
        save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train CLIP model on match/mismatch dataset.
        
        Args:
            train_csv: Path to training CSV file
            val_csv: Optional path to validation CSV file
            epochs: Number of training epochs
            batch_size: Batch size for training
            output_dir: Directory to save checkpoints
            image_base_path: Base path for image files in CSV
            save_best_only: If True, only save checkpoint with best validation loss
            
        Returns:
            dict: Training history with 'train_loss' and 'val_loss' lists
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load datasets
        logger.info(f"Loading training dataset from: {train_csv}")
        train_dataset = CLIPMatchDataset(train_csv, self.processor, image_base_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = None
        if val_csv:
            logger.info(f"Loading validation dataset from: {val_csv}")
            val_dataset = CLIPMatchDataset(val_csv, self.processor, image_base_path)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Validate
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                history['val_loss'].append(val_loss)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if save_best_only:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(output_dir, epoch, is_best=True)
                        logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
                else:
                    self._save_checkpoint(output_dir, epoch, is_best=False)
            else:
                # No validation set, save every epoch
                self._save_checkpoint(output_dir, epoch, is_best=False)
        
        # Save final model
        final_path = os.path.join(output_dir, 'final_model')
        self._save_checkpoint(final_path, epochs - 1, is_best=False)
        logger.info(f"Training complete. Final model saved to: {final_path}")
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=False
            )
            
            # Compute similarity scores
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_embeds * text_embeds).sum(dim=-1)
            
            # Binary cross-entropy loss
            # similarity ranges from -1 to 1, convert to 0 to 1 for BCE
            similarity_normalized = (similarity + 1) / 2
            loss = torch.nn.functional.binary_cross_entropy(
                similarity_normalized,
                labels,
                reduction='mean'
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            float: Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=False
                )
                
                # Compute similarity scores
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_embeds * text_embeds).sum(dim=-1)
                
                # Binary cross-entropy loss
                similarity_normalized = (similarity + 1) / 2
                loss = torch.nn.functional.binary_cross_entropy(
                    similarity_normalized,
                    labels,
                    reduction='mean'
                )
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(
        self,
        output_dir: str,
        epoch: int,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint
            epoch: Current epoch number
            is_best: If True, saves as 'best_model' instead of epoch number
        """
        if is_best:
            save_path = os.path.join(output_dir, 'best_model')
        else:
            save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}')
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and processor
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        logger.info(f"Checkpoint saved to: {save_path}")


def validate_dataset(csv_path: str, image_base_path: Optional[str] = None) -> Dict[str, any]:
    """
    Validate dataset CSV and check if all images exist.
    
    Args:
        csv_path: Path to dataset CSV file
        image_base_path: Optional base path for image files
        
    Returns:
        dict: Validation report with statistics and errors
    """
    logger.info(f"Validating dataset: {csv_path}")
    
    report = {
        'valid': True,
        'total_samples': 0,
        'match_samples': 0,
        'mismatch_samples': 0,
        'missing_images': [],
        'invalid_labels': [],
        'errors': []
    }
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        report['total_samples'] = len(df)
        
        # Check required columns
        required_cols = ['image_path', 'text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report['valid'] = False
            report['errors'].append(f"Missing columns: {missing_cols}")
            return report
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            report['valid'] = False
            report['errors'].append(f"Null values found: {null_counts.to_dict()}")
        
        # Validate labels
        invalid_labels = df[~df['label'].isin([0, 1])].index.tolist()
        if invalid_labels:
            report['valid'] = False
            report['invalid_labels'] = invalid_labels
            report['errors'].append(f"Invalid labels at rows: {invalid_labels}")
        
        # Count label distribution
        report['match_samples'] = int((df['label'] == 1).sum())
        report['mismatch_samples'] = int((df['label'] == 0).sum())
        
        # Check if images exist
        base_path = image_base_path or ""
        for idx, row in df.iterrows():
            img_path = os.path.join(base_path, row['image_path'])
            if not os.path.exists(img_path):
                report['missing_images'].append(img_path)
        
        if report['missing_images']:
            report['valid'] = False
            report['errors'].append(f"Missing {len(report['missing_images'])} images")
        
        # Check label balance
        if report['match_samples'] > 0 and report['mismatch_samples'] > 0:
            ratio = report['match_samples'] / report['mismatch_samples']
            if ratio > 5 or ratio < 0.2:
                report['errors'].append(
                    f"Imbalanced dataset: match/mismatch ratio = {ratio:.2f}"
                )
        
    except Exception as e:
        report['valid'] = False
        report['errors'].append(f"Error reading CSV: {str(e)}")
        logger.error(f"Dataset validation error: {str(e)}", exc_info=True)
    
    logger.info(f"Validation complete. Valid: {report['valid']}")
    
    return report
