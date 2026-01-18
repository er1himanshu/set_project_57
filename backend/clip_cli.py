#!/usr/bin/env python3
"""
CLI tool for CLIP model operations including training, evaluation, and inference.
"""

import argparse
import sys
import os

# Add parent directory to path to import training module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_clip import CLIPFineTuner
from app.services.clip_service import get_clip_service, analyze_image_text_match


def train_command(args):
    """Execute training command."""
    print("=" * 60)
    print("CLIP Model Fine-Tuning")
    print("=" * 60)
    print(f"Training data: {args.train_csv}")
    print(f"Validation data: {args.val_csv or 'None'}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    fine_tuner = CLIPFineTuner(model_name=args.model)
    fine_tuner.train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_steps=args.save_steps,
        root_dir=args.root_dir
    )
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {args.output}")


def test_command(args):
    """Execute test command for single image-text pair."""
    print("=" * 60)
    print("CLIP Image-Text Similarity Test")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Text: {args.text}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)
    
    result = analyze_image_text_match(
        image_path=args.image,
        description=args.text,
        threshold=args.threshold
    )
    
    print("\nResults:")
    print(f"  Similarity Score: {result['similarity_score']:.4f}")
    print(f"  Is Match: {result['is_match']}")
    print(f"  Status: {result['status']}")
    
    if result['is_match']:
        print("\n✅ Image and text MATCH")
    else:
        print("\n❌ Image and text MISMATCH")


def batch_test_command(args):
    """Execute batch testing from CSV file."""
    import pandas as pd
    from tqdm import tqdm
    
    print("=" * 60)
    print("CLIP Batch Testing")
    print("=" * 60)
    print(f"Input CSV: {args.csv}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    
    if 'image_path' not in df.columns or 'text' not in df.columns:
        print("❌ Error: CSV must contain 'image_path' and 'text' columns")
        return
    
    # Process each row
    results = []
    correct = 0
    total = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_path = row['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(args.root_dir or os.path.dirname(args.csv), image_path)
        
        text = row['text']
        expected_label = row.get('label', None)
        
        result = analyze_image_text_match(
            image_path=image_path,
            description=text,
            threshold=args.threshold
        )
        
        predicted = 'match' if result['is_match'] else 'mismatch'
        
        results.append({
            'image_path': row['image_path'],
            'text': text,
            'similarity_score': result['similarity_score'],
            'predicted': predicted,
            'expected': expected_label,
            'status': result['status']
        })
        
        # Calculate accuracy if labels provided
        if expected_label:
            if predicted == str(expected_label).lower():
                correct += 1
            total += 1
    
    # Save results
    output_csv = args.output or args.csv.replace('.csv', '_results.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Batch testing completed!")
    print(f"Results saved to: {output_csv}")
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")


def info_command(args):
    """Display information about current CLIP configuration."""
    from app.config import (
        CLIP_MODEL_NAME, 
        CLIP_SIMILARITY_THRESHOLD, 
        CLIP_DEVICE,
        CLIP_ZERO_SHOT_LABELS
    )
    
    print("=" * 60)
    print("CLIP Configuration")
    print("=" * 60)
    print(f"Model: {CLIP_MODEL_NAME}")
    print(f"Similarity Threshold: {CLIP_SIMILARITY_THRESHOLD}")
    print(f"Device: {CLIP_DEVICE}")
    print(f"Zero-shot Labels: {CLIP_ZERO_SHOT_LABELS or 'None configured'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for CLIP-based image-text matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a CLIP model
  python clip_cli.py train --train-csv datasets/train.csv --val-csv datasets/val.csv
  
  # Test a single image-text pair
  python clip_cli.py test --image product.jpg --text "Red cotton shirt"
  
  # Batch test from CSV
  python clip_cli.py batch-test --csv test_data.csv
  
  # Show current configuration
  python clip_cli.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Fine-tune CLIP model')
    train_parser.add_argument('--train-csv', required=True, help='Training data CSV')
    train_parser.add_argument('--val-csv', help='Validation data CSV')
    train_parser.add_argument('--model', default='openai/clip-vit-base-patch32',
                            help='Pretrained model name')
    train_parser.add_argument('--output', default='./fine_tuned_clip',
                            help='Output directory for fine-tuned model')
    train_parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    train_parser.add_argument('--save-steps', type=int, default=100,
                            help='Save checkpoint every N steps')
    train_parser.add_argument('--root-dir', help='Root directory for image paths')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test single image-text pair')
    test_parser.add_argument('--image', required=True, help='Path to image file')
    test_parser.add_argument('--text', required=True, help='Text description')
    test_parser.add_argument('--threshold', type=float, default=0.25,
                           help='Similarity threshold (0-1)')
    
    # Batch test command
    batch_parser = subparsers.add_parser('batch-test', help='Batch test from CSV')
    batch_parser.add_argument('--csv', required=True, help='Input CSV file')
    batch_parser.add_argument('--threshold', type=float, default=0.25,
                            help='Similarity threshold (0-1)')
    batch_parser.add_argument('--output', help='Output CSV file for results')
    batch_parser.add_argument('--root-dir', help='Root directory for image paths')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show CLIP configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'batch-test':
        batch_test_command(args)
    elif args.command == 'info':
        info_command(args)


if __name__ == "__main__":
    main()
