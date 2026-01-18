#!/usr/bin/env python3
"""
Training script for fine-tuning CLIP models on image-text match/mismatch datasets.

Usage:
    python train_clip.py --train-csv path/to/train.csv --val-csv path/to/val.csv
    
Example:
    python train_clip.py \
        --train-csv data/train.csv \
        --val-csv data/val.csv \
        --epochs 5 \
        --batch-size 16 \
        --learning-rate 5e-6 \
        --output-dir ./clip_checkpoints
"""

import argparse
import sys
import os
import logging

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.clip_trainer import CLIPTrainer, validate_dataset
from app.config import (
    CLIP_MODEL_NAME,
    CLIP_TRAIN_BATCH_SIZE,
    CLIP_TRAIN_LEARNING_RATE,
    CLIP_TRAIN_EPOCHS,
    CLIP_CHECKPOINT_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP model on image-text match/mismatch dataset"
    )
    
    # Required arguments
    parser.add_argument(
        '--train-csv',
        type=str,
        required=True,
        help='Path to training CSV file (image_path,text,label)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--val-csv',
        type=str,
        default=None,
        help='Path to validation CSV file (optional)'
    )
    
    parser.add_argument(
        '--image-base-path',
        type=str,
        default=None,
        help='Base path for image files (prepended to paths in CSV)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=CLIP_TRAIN_EPOCHS,
        help=f'Number of training epochs (default: {CLIP_TRAIN_EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=CLIP_TRAIN_BATCH_SIZE,
        help=f'Batch size for training (default: {CLIP_TRAIN_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=CLIP_TRAIN_LEARNING_RATE,
        help=f'Learning rate (default: {CLIP_TRAIN_LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=CLIP_MODEL_NAME,
        help=f'Base CLIP model to fine-tune (default: {CLIP_MODEL_NAME})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=CLIP_CHECKPOINT_DIR,
        help=f'Directory to save checkpoints (default: {CLIP_CHECKPOINT_DIR})'
    )
    
    parser.add_argument(
        '--save-all-checkpoints',
        action='store_true',
        help='Save checkpoint after every epoch (default: save best only)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the dataset without training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to train on (default: auto-detect)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("CLIP Fine-tuning Script")
    logger.info("=" * 60)
    
    # Validate training dataset
    logger.info("\nValidating training dataset...")
    train_report = validate_dataset(args.train_csv, args.image_base_path)
    
    logger.info(f"Training dataset: {args.train_csv}")
    logger.info(f"  Total samples: {train_report['total_samples']}")
    logger.info(f"  Match samples: {train_report['match_samples']}")
    logger.info(f"  Mismatch samples: {train_report['mismatch_samples']}")
    
    if not train_report['valid']:
        logger.error("Training dataset validation failed!")
        for error in train_report['errors']:
            logger.error(f"  - {error}")
        if train_report['missing_images']:
            logger.error(f"  Missing images (first 10):")
            for img in train_report['missing_images'][:10]:
                logger.error(f"    - {img}")
        sys.exit(1)
    
    logger.info("✓ Training dataset is valid")
    
    # Validate validation dataset if provided
    if args.val_csv:
        logger.info("\nValidating validation dataset...")
        val_report = validate_dataset(args.val_csv, args.image_base_path)
        
        logger.info(f"Validation dataset: {args.val_csv}")
        logger.info(f"  Total samples: {val_report['total_samples']}")
        logger.info(f"  Match samples: {val_report['match_samples']}")
        logger.info(f"  Mismatch samples: {val_report['mismatch_samples']}")
        
        if not val_report['valid']:
            logger.error("Validation dataset validation failed!")
            for error in val_report['errors']:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        logger.info("✓ Validation dataset is valid")
    
    # Exit if validate-only mode
    if args.validate_only:
        logger.info("\nValidation complete. Exiting (--validate-only mode).")
        return
    
    # Initialize trainer
    logger.info("\nInitializing CLIP trainer...")
    logger.info(f"  Base model: {args.model_name}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {args.device or 'auto-detect'}")
    
    try:
        trainer = CLIPTrainer(
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            device=args.device
        )
    except Exception as e:
        logger.error("Failed to initialize CLIP trainer!")
        logger.error(f"Error: {str(e)}")
        logger.error("\nPossible causes:")
        logger.error("  - Model not available or incorrect model name")
        logger.error("  - Missing dependencies (torch, transformers)")
        logger.error("  - Insufficient memory or CUDA issues")
        logger.error(f"\nTried to load: {args.model_name}")
        logger.error("Verify the model name is correct and dependencies are installed")
        sys.exit(1)
    
    # Train model
    logger.info("\nStarting training...")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Save strategy: {'all checkpoints' if args.save_all_checkpoints else 'best only'}")
    
    try:
        history = trainer.train(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            image_base_path=args.image_base_path,
            save_best_only=not args.save_all_checkpoints
        )
        
        # Print training summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"\nFinal Training Loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
            logger.info(f"Best Validation Loss: {min(history['val_loss']):.4f}")
        
        logger.info(f"\nModel saved to: {args.output_dir}")
        logger.info("\nTo use the fine-tuned model, set environment variable:")
        logger.info(f"  export CLIP_MODEL_PATH={os.path.join(args.output_dir, 'best_model')}")
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
