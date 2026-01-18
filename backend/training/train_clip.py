"""
Training script for fine-tuning CLIP on image-text mismatch detection.

This script fine-tunes CLIP model on a labeled dataset to improve
performance on specific domain (e.g., ecommerce product images).

Usage:
    python train_clip.py --train_csv data/train.csv --val_csv data/val.csv --output_dir ./fine_tuned_clip

CSV Format:
    image_path,text,label
    images/product1.jpg,"Red leather handbag",1
    images/product2.jpg,"Blue cotton shirt",0
    
    where label: 1 = match, 0 = mismatch
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import json
from datetime import datetime

from dataset_loader import create_dataloaders, validate_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIPFineTuner:
    """Fine-tuner for CLIP models on binary classification task."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (auto-detect if None)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.projection_dim  # Typically 512 for base
        
        # Add classification head on top of CLIP
        # Input: concatenated [image_embed, text_embed, similarity]
        # Size: embedding_dim + embedding_dim + 1
        input_dim = self.embedding_dim * 2 + 1
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary classification (match/mismatch)
        ).to(self.device)
        
        logger.info(f"Classification head input dim: {input_dim}")
        
        # Setup optimizer
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.classification_head.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.classification_head.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move batch to device
            inputs = {
                k: v.to(self.device) 
                for k, v in batch.items() 
                if k != 'labels'
            }
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get CLIP embeddings
            outputs = self.model(**inputs)
            
            # Get normalized embeddings
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_embeds * text_embeds).sum(dim=-1, keepdim=True)
            
            # Concatenate features
            features = torch.cat([image_embeds, text_embeds, similarity], dim=-1)
            
            # Classification
            logits = self.classification_head(features)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        self.train_losses.append(avg_loss)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, val_loader, epoch: int):
        """Validate the model."""
        self.model.eval()
        self.classification_head.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
            # Move batch to device
            inputs = {
                k: v.to(self.device) 
                for k, v in batch.items() 
                if k != 'labels'
            }
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Get features
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            similarity = (image_embeds * text_embeds).sum(dim=-1, keepdim=True)
            features = torch.cat([image_embeds, text_embeds, similarity], dim=-1)
            
            # Classification
            logits = self.classification_head(features)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def save_model(self, output_dir: str, epoch: int = None):
        """Save fine-tuned model."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CLIP model
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        # Save classification head
        head_path = os.path.join(output_dir, 'classification_head.pt')
        torch.save(self.classification_head.state_dict(), head_path)
        
        # Save training stats
        stats = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'epoch': epoch
        }
        stats_path = os.path.join(output_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for image-text mismatch detection")
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='Path to validation CSV file (optional)')
    parser.add_argument('--image_base_path', type=str, default='',
                        help='Base path to prepend to image paths in CSV')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32',
                        help='HuggingFace model name')
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_clip',
                        help='Directory to save fine-tuned model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    # Validation arguments
    parser.add_argument('--validate_data', action='store_true',
                        help='Validate dataset before training')
    
    args = parser.parse_args()
    
    # Validate dataset if requested
    if args.validate_data:
        logger.info("Validating training dataset...")
        train_stats = validate_dataset(args.train_csv, args.image_base_path)
        logger.info(f"Training dataset stats: {train_stats}")
        
        if args.val_csv:
            logger.info("Validating validation dataset...")
            val_stats = validate_dataset(args.val_csv, args.image_base_path)
            logger.info(f"Validation dataset stats: {val_stats}")
    
    # Create fine-tuner
    fine_tuner = CLIPFineTuner(
        model_name=args.model_name,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        processor=fine_tuner.processor,
        image_base_path=args.image_base_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup learning rate scheduler
    scheduler = CosineAnnealingLR(
        fine_tuner.optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    # Training loop
    best_val_acc = 0.0
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = fine_tuner.train_epoch(train_loader, epoch)
        
        # Validate
        if val_loader:
            val_loss, val_acc = fine_tuner.validate(val_loader, epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_dir = os.path.join(args.output_dir, 'best_model')
                fine_tuner.save_model(best_model_dir, epoch)
                logger.info(f"Saved best model (val_acc: {val_acc:.2f}%)")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}')
        fine_tuner.save_model(checkpoint_dir, epoch)
    
    # Save final model
    final_model_dir = os.path.join(args.output_dir, 'final_model')
    fine_tuner.save_model(final_model_dir, args.epochs)
    logger.info("Training complete!")
    
    # Print summary
    logger.info(f"\nTraining Summary:")
    logger.info(f"  Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"  Final train loss: {fine_tuner.train_losses[-1]:.4f}")
    if fine_tuner.val_losses:
        logger.info(f"  Final val loss: {fine_tuner.val_losses[-1]:.4f}")
    logger.info(f"  Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
