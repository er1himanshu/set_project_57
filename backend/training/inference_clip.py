"""
Inference script for CLIP mismatch detection.

This script can be used standalone to check image-text similarity
using either pre-trained or fine-tuned CLIP models.

Usage:
    # Single image inference
    python inference_clip.py --image path/to/image.jpg --text "Product description" --model ./fine_tuned_clip/best_model
    
    # Batch inference from CSV
    python inference_clip.py --csv data/test.csv --model ./fine_tuned_clip/best_model --output results.csv
"""

import os
import argparse
import logging
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIPInference:
    """Inference class for CLIP mismatch detection."""
    
    def __init__(
        self,
        model_path: str = "openai/clip-vit-base-patch32",
        device: str = None,
        threshold: float = 0.6
    ):
        """
        Initialize inference.
        
        Args:
            model_path: Path to fine-tuned model or HuggingFace model name
            device: Device to use (auto-detect if None)
            threshold: Threshold for match/mismatch decision
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model from: {model_path}")
        
        # Load model and processor
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def compute_similarity(
        self,
        image_path: str,
        text: str
    ) -> float:
        """
        Compute similarity score between image and text.
        
        Args:
            image_path: Path to image file
            text: Text description
            
        Returns:
            Similarity score (0-1)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute similarity
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            similarity_score = probs[0][0].item()
        
        return similarity_score
    
    def predict(
        self,
        image_path: str,
        text: str,
        threshold: float = None
    ) -> dict:
        """
        Predict match/mismatch for image-text pair.
        
        Args:
            image_path: Path to image file
            text: Text description
            threshold: Custom threshold (uses instance threshold if None)
            
        Returns:
            Dictionary with prediction results
        """
        threshold = threshold if threshold is not None else self.threshold
        
        similarity_score = self.compute_similarity(image_path, text)
        is_match = similarity_score >= threshold
        
        return {
            'image_path': image_path,
            'text': text,
            'similarity_score': similarity_score,
            'threshold': threshold,
            'is_match': is_match,
            'prediction': 'match' if is_match else 'mismatch'
        }
    
    def batch_predict(
        self,
        csv_path: str,
        image_base_path: str = '',
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Batch prediction from CSV file.
        
        Args:
            csv_path: Path to CSV with image_path and text columns
            image_base_path: Base path for images
            threshold: Custom threshold
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        if 'image_path' not in df.columns or 'text' not in df.columns:
            raise ValueError("CSV must contain 'image_path' and 'text' columns")
        
        results = []
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            image_path = os.path.join(image_base_path, row['image_path'])
            text = row['text']
            
            try:
                result = self.predict(image_path, text, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'text': text,
                    'similarity_score': None,
                    'threshold': threshold or self.threshold,
                    'is_match': None,
                    'prediction': 'error',
                    'error': str(e)
                })
        
        return pd.DataFrame(results)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="CLIP inference for image-text mismatch detection"
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32',
                        help='Path to fine-tuned model or HuggingFace model name')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold for match/mismatch decision (default: 0.6)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    # Single image inference
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--text', type=str, default=None,
                        help='Text description for single image inference')
    
    # Batch inference
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV for batch inference')
    parser.add_argument('--image_base_path', type=str, default='',
                        help='Base path for images in CSV')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV path for batch inference')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = CLIPInference(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold
    )
    
    # Single image inference
    if args.image and args.text:
        logger.info(f"Running inference on single image: {args.image}")
        result = inference.predict(args.image, args.text)
        
        # Print result
        print("\n" + "="*50)
        print("CLIP Mismatch Detection Result")
        print("="*50)
        print(f"Image: {result['image_path']}")
        print(f"Text: {result['text']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Threshold: {result['threshold']}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Is Match: {result['is_match']}")
        print("="*50 + "\n")
        
        # Save to JSON
        output_json = 'result.json'
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {output_json}")
    
    # Batch inference
    elif args.csv:
        logger.info(f"Running batch inference on CSV: {args.csv}")
        results_df = inference.batch_predict(
            args.csv,
            args.image_base_path,
            args.threshold
        )
        
        # Save results
        results_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        if 'prediction' in results_df.columns:
            print("\n" + "="*50)
            print("Batch Inference Summary")
            print("="*50)
            print(f"Total samples: {len(results_df)}")
            print(f"Predictions:\n{results_df['prediction'].value_counts()}")
            
            valid_scores = results_df['similarity_score'].dropna()
            if not valid_scores.empty:
                print(f"\nSimilarity Score Statistics:")
                print(f"  Mean: {valid_scores.mean():.4f}")
                print(f"  Std: {valid_scores.std():.4f}")
                print(f"  Min: {valid_scores.min():.4f}")
                print(f"  Max: {valid_scores.max():.4f}")
            print("="*50 + "\n")
    
    else:
        parser.print_help()
        print("\nError: Please provide either (--image and --text) for single inference or --csv for batch inference")


if __name__ == '__main__':
    main()
