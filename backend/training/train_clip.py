"""
Fine-tuning script for CLIP model on custom image-text pairs.

This script allows fine-tuning a pre-trained CLIP model on your own dataset
of product images and descriptions to improve image-text matching accuracy.
"""

import os
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AdamW
from PIL import Image
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTextDataset(Dataset):
    """
    Dataset for loading image-text pairs from a CSV file.
    
    CSV Format:
    image_path,text,label
    /path/to/image1.jpg,"Red cotton shirt",match
    /path/to/image2.jpg,"Blue leather shoes",match
    /path/to/image3.jpg,"Green cotton shirt",mismatch
    
    Where:
    - image_path: Path to the image file (can be absolute or relative to dataset root)
    - text: Product description text
    - label: "match" or "mismatch" indicating if image and text correspond
    """
    
    def __init__(self, csv_path: str, processor, root_dir: str = None):
        """
        Args:
            csv_path: Path to CSV file with image-text pairs
            processor: CLIP processor for preprocessing
            root_dir: Root directory for relative image paths (optional)
        """
        self.data = pd.read_csv(csv_path)
        self.processor = processor
        self.root_dir = root_dir or os.path.dirname(csv_path)
        
        logger.info(f"Loaded {len(self.data)} samples from {csv_path}")
        
        # Validate data
        required_columns = ['image_path', 'text', 'label']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert labels to binary (1 for match, 0 for mismatch)
        self.data['label'] = self.data['label'].apply(
            lambda x: 1 if str(x).lower() == 'match' else 0
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.root_dir, image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        text = str(row['text'])
        label = int(row['label'])
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        
        return inputs


class CLIPFineTuner:
    """Fine-tuning wrapper for CLIP models."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        """
        Args:
            model_name: Hugging Face model name/path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model.to(self.device)
        
        # Loss function for contrastive learning
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(
        self,
        train_csv: str,
        val_csv: str = None,
        output_dir: str = "./fine_tuned_clip",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-6,
        save_steps: int = 100,
        root_dir: str = None
    ):
        """
        Fine-tune the CLIP model on custom dataset.
        
        Args:
            train_csv: Path to training data CSV
            val_csv: Path to validation data CSV (optional)
            output_dir: Directory to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            save_steps: Save checkpoint every N steps
            root_dir: Root directory for relative image paths
        """
        # Create datasets
        train_dataset = ImageTextDataset(train_csv, self.processor, root_dir)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = None
        if val_csv:
            val_dataset = ImageTextDataset(val_csv, self.processor, root_dir)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get image and text embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_embeds * text_embeds).sum(dim=-1)
                
                # Compute contrastive loss
                # Labels: 1 for match, 0 for mismatch
                # We want high similarity for match (label=1), low for mismatch (label=0)
                loss = self.criterion(similarity, labels.float())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    self.save_model(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            avg_train_loss = epoch_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_dir = os.path.join(output_dir, "best_model")
                    self.save_model(best_model_dir)
                    logger.info(f"Saved best model to {best_model_dir}")
        
        # Save final model
        final_model_dir = os.path.join(output_dir, "final_model")
        self.save_model(final_model_dir)
        logger.info(f"Training complete! Final model saved to {final_model_dir}")
    
    def evaluate(self, val_loader):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                
                outputs = self.model(**batch)
                
                # Get image and text embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_embeds * text_embeds).sum(dim=-1)
                
                # Compute loss
                loss = self.criterion(similarity, labels.float())
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def save_model(self, output_dir: str):
        """Save model and processor to directory."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP model on custom image-text pairs"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Path to validation data CSV file (optional)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Pretrained CLIP model name from Hugging Face"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine_tuned_clip",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory for relative image paths in CSV"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    
    if args.val_csv and not os.path.exists(args.val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {args.val_csv}")
    
    # Create fine-tuner and train
    fine_tuner = CLIPFineTuner(model_name=args.model_name)
    
    fine_tuner.train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        root_dir=args.root_dir
    )


if __name__ == "__main__":
    main()
