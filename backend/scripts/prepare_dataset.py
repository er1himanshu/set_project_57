#!/usr/bin/env python3
"""
Dataset preparation and validation script for CLIP training.

This script helps prepare and validate datasets for CLIP fine-tuning.
It can check dataset integrity, split data, and generate sample datasets.

Usage:
    python prepare_dataset.py --csv path/to/dataset.csv --validate
    python prepare_dataset.py --csv path/to/dataset.csv --split 0.8 --output-dir ./data
"""

import argparse
import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.clip_trainer import validate_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare and validate CLIP training datasets"
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to dataset CSV file'
    )
    
    parser.add_argument(
        '--image-base-path',
        type=str,
        default=None,
        help='Base path for image files'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the dataset'
    )
    
    parser.add_argument(
        '--split',
        type=float,
        default=None,
        help='Train/validation split ratio (e.g., 0.8 for 80/20 split)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for split datasets'
    )
    
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle data before splitting'
    )
    
    parser.add_argument(
        '--create-sample',
        type=int,
        default=None,
        help='Create a sample dataset with N examples'
    )
    
    return parser.parse_args()


def split_dataset(
    csv_path: str,
    train_ratio: float,
    output_dir: str,
    shuffle: bool = True
) -> tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        csv_path: Path to input CSV
        train_ratio: Ratio of training data (0-1)
        output_dir: Directory to save split files
        shuffle: Whether to shuffle before splitting
        
    Returns:
        tuple: (train_csv_path, val_csv_path)
    """
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info("Dataset shuffled")
    
    # Calculate split point
    train_size = int(len(df) * train_ratio)
    
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save split datasets
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    logger.info(f"Saved train set to: {train_path}")
    logger.info(f"Saved validation set to: {val_path}")
    
    return train_path, val_path


def create_sample_dataset(output_path: str, n_samples: int):
    """
    Create a sample dataset for testing.
    
    Args:
        output_path: Path to save sample CSV
        n_samples: Number of samples to create
    """
    logger.info(f"Creating sample dataset with {n_samples} examples")
    
    # Create sample data
    data = {
        'image_path': [],
        'text': [],
        'label': []
    }
    
    # Example entries (users should replace with actual data)
    sample_entries = [
        ('path/to/red_shoes.jpg', 'red running shoes', 1),
        ('path/to/red_shoes.jpg', 'blue formal dress', 0),
        ('path/to/laptop.jpg', 'silver laptop computer', 1),
        ('path/to/laptop.jpg', 'wooden chair', 0),
        ('path/to/handbag.jpg', 'leather handbag', 1),
        ('path/to/handbag.jpg', 'electronic device', 0),
    ]
    
    # Repeat to reach n_samples
    for i in range(n_samples):
        entry = sample_entries[i % len(sample_entries)]
        data['image_path'].append(entry[0])
        data['text'].append(entry[1])
        data['label'].append(entry[2])
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample dataset saved to: {output_path}")
    logger.info("⚠️  Note: This is a template. Replace with actual image paths and descriptions.")


def print_dataset_stats(csv_path: str, image_base_path: str = None):
    """Print detailed dataset statistics."""
    df = pd.read_csv(csv_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Match samples (label=1): {(df['label'] == 1).sum()}")
    logger.info(f"Mismatch samples (label=0): {(df['label'] == 0).sum()}")
    
    if 'label' in df.columns:
        match_ratio = (df['label'] == 1).sum() / len(df)
        logger.info(f"Match ratio: {match_ratio:.2%}")
    
    logger.info(f"\nText length statistics:")
    df['text_length'] = df['text'].str.len()
    logger.info(f"  Mean: {df['text_length'].mean():.1f} characters")
    logger.info(f"  Min: {df['text_length'].min()} characters")
    logger.info(f"  Max: {df['text_length'].max()} characters")
    
    logger.info(f"\nSample entries:")
    for i, row in df.head(3).iterrows():
        logger.info(f"  [{i}] {row['image_path']} | {row['text'][:50]}... | label={row['label']}")


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("CLIP Dataset Preparation Tool")
    logger.info("=" * 60)
    
    # Create sample dataset
    if args.create_sample:
        create_sample_dataset(args.csv, args.create_sample)
        return
    
    # Validate dataset
    if args.validate:
        logger.info("\nValidating dataset...")
        report = validate_dataset(args.csv, args.image_base_path)
        
        if report['valid']:
            logger.info("✓ Dataset is valid!")
            print_dataset_stats(args.csv, args.image_base_path)
        else:
            logger.error("✗ Dataset validation failed!")
            for error in report['errors']:
                logger.error(f"  - {error}")
            
            if report['missing_images']:
                logger.error(f"\nMissing images (first 10):")
                for img in report['missing_images'][:10]:
                    logger.error(f"  - {img}")
            
            sys.exit(1)
    
    # Split dataset
    if args.split:
        if not 0 < args.split < 1:
            logger.error("Split ratio must be between 0 and 1")
            sys.exit(1)
        
        logger.info(f"\nSplitting dataset (ratio: {args.split})...")
        train_path, val_path = split_dataset(
            args.csv,
            args.split,
            args.output_dir,
            args.shuffle
        )
        
        # Validate split datasets
        logger.info("\nValidating train set...")
        train_report = validate_dataset(train_path, args.image_base_path)
        if not train_report['valid']:
            logger.error("Train set validation failed!")
            sys.exit(1)
        
        logger.info("\nValidating validation set...")
        val_report = validate_dataset(val_path, args.image_base_path)
        if not val_report['valid']:
            logger.error("Validation set validation failed!")
            sys.exit(1)
        
        logger.info("\n✓ Split complete and validated!")
    
    if not args.validate and not args.split and not args.create_sample:
        logger.info("\nNo action specified. Use --validate, --split, or --create-sample")
        logger.info("Run with --help for usage information")


if __name__ == '__main__':
    main()
