"""
Basic tests for CLIP integration to verify implementation works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.services.clip_service import (
            CLIPService,
            get_clip_service,
            analyze_image_text_match
        )
        print("  ✓ clip_service imports successful")
    except ImportError as e:
        print(f"  ✗ Failed to import clip_service: {e}")
        return False
    
    try:
        from app.services.image_quality import analyze_image
        print("  ✓ image_quality imports successful")
    except ImportError as e:
        print(f"  ✗ Failed to import image_quality: {e}")
        return False
    
    try:
        from app.config import (
            CLIP_MODEL_NAME,
            CLIP_SIMILARITY_THRESHOLD,
            CLIP_DEVICE,
            CLIP_ZERO_SHOT_LABELS
        )
        print("  ✓ config imports successful")
        print(f"    - Model: {CLIP_MODEL_NAME}")
        print(f"    - Threshold: {CLIP_SIMILARITY_THRESHOLD}")
        print(f"    - Device: {CLIP_DEVICE}")
    except ImportError as e:
        print(f"  ✗ Failed to import config: {e}")
        return False
    
    try:
        from app.models import ImageResult
        print("  ✓ models imports successful")
    except ImportError as e:
        print(f"  ✗ Failed to import models: {e}")
        return False
    
    try:
        from app.schemas import ImageResultSchema
        print("  ✓ schemas imports successful")
    except ImportError as e:
        print(f"  ✗ Failed to import schemas: {e}")
        return False
    
    return True


def test_model_fields():
    """Test that database models have the new CLIP fields."""
    print("\nTesting database model fields...")
    
    try:
        from app.models import ImageResult
        from sqlalchemy import inspect
        
        # Get model columns
        mapper = inspect(ImageResult)
        columns = [c.key for c in mapper.columns]
        
        # Check for CLIP fields
        required_fields = [
            'clip_similarity_score',
            'clip_match_status',
            'clip_is_match'
        ]
        
        for field in required_fields:
            if field in columns:
                print(f"  ✓ Field '{field}' exists in ImageResult model")
            else:
                print(f"  ✗ Field '{field}' missing from ImageResult model")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking model fields: {e}")
        return False


def test_schema_fields():
    """Test that Pydantic schemas have the new CLIP fields."""
    print("\nTesting schema fields...")
    
    try:
        from app.schemas import ImageResultSchema
        
        # Check for CLIP fields in schema
        required_fields = [
            'clip_similarity_score',
            'clip_match_status',
            'clip_is_match'
        ]
        
        schema_fields = ImageResultSchema.model_fields.keys()
        
        for field in required_fields:
            if field in schema_fields:
                print(f"  ✓ Field '{field}' exists in ImageResultSchema")
            else:
                print(f"  ✗ Field '{field}' missing from ImageResultSchema")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking schema fields: {e}")
        return False


def test_config_values():
    """Test that configuration has proper values."""
    print("\nTesting configuration values...")
    
    try:
        from app.config import (
            CLIP_MODEL_NAME,
            CLIP_SIMILARITY_THRESHOLD,
            CLIP_DEVICE,
            CLIP_ZERO_SHOT_LABELS
        )
        
        # Test model name
        if CLIP_MODEL_NAME and isinstance(CLIP_MODEL_NAME, str):
            print(f"  ✓ CLIP_MODEL_NAME is valid: {CLIP_MODEL_NAME}")
        else:
            print(f"  ✗ CLIP_MODEL_NAME is invalid: {CLIP_MODEL_NAME}")
            return False
        
        # Test threshold
        if 0 <= CLIP_SIMILARITY_THRESHOLD <= 1:
            print(f"  ✓ CLIP_SIMILARITY_THRESHOLD is valid: {CLIP_SIMILARITY_THRESHOLD}")
        else:
            print(f"  ✗ CLIP_SIMILARITY_THRESHOLD out of range: {CLIP_SIMILARITY_THRESHOLD}")
            return False
        
        # Test device
        if CLIP_DEVICE in ["cpu", "cuda"]:
            print(f"  ✓ CLIP_DEVICE is valid: {CLIP_DEVICE}")
        else:
            print(f"  ✗ CLIP_DEVICE is invalid: {CLIP_DEVICE}")
            return False
        
        # Test labels (can be empty)
        if isinstance(CLIP_ZERO_SHOT_LABELS, list):
            print(f"  ✓ CLIP_ZERO_SHOT_LABELS is valid list ({len(CLIP_ZERO_SHOT_LABELS)} items)")
        else:
            print(f"  ✗ CLIP_ZERO_SHOT_LABELS is not a list")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking config values: {e}")
        return False


def test_cli_script():
    """Test that CLI script exists and is executable."""
    print("\nTesting CLI script...")
    
    cli_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'clip_cli.py')
    
    if os.path.exists(cli_path):
        print(f"  ✓ CLI script exists: {cli_path}")
    else:
        print(f"  ✗ CLI script not found: {cli_path}")
        return False
    
    if os.access(cli_path, os.X_OK):
        print(f"  ✓ CLI script is executable")
    else:
        print(f"  ⚠ CLI script is not executable (this is OK)")
    
    # Test syntax
    try:
        with open(cli_path, 'r') as f:
            compile(f.read(), cli_path, 'exec')
        print(f"  ✓ CLI script syntax is valid")
    except SyntaxError as e:
        print(f"  ✗ CLI script has syntax error: {e}")
        return False
    
    return True


def test_training_script():
    """Test that training script exists."""
    print("\nTesting training script...")
    
    train_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'training', 'train_clip.py'
    )
    
    if os.path.exists(train_path):
        print(f"  ✓ Training script exists: {train_path}")
    else:
        print(f"  ✗ Training script not found: {train_path}")
        return False
    
    # Test syntax
    try:
        with open(train_path, 'r') as f:
            compile(f.read(), train_path, 'exec')
        print(f"  ✓ Training script syntax is valid")
    except SyntaxError as e:
        print(f"  ✗ Training script has syntax error: {e}")
        return False
    
    return True


def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    docs = [
        ('docs/CLIP_INTEGRATION.md', 'CLIP Integration Guide'),
        ('backend/training/datasets/README.md', 'Dataset Format Guide'),
        ('backend/examples/README.md', 'Examples README')
    ]
    
    all_exist = True
    for doc_path, doc_name in docs:
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            doc_path
        )
        if os.path.exists(full_path):
            print(f"  ✓ {doc_name} exists")
        else:
            print(f"  ✗ {doc_name} not found: {full_path}")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("CLIP Integration - Basic Tests")
    print("=" * 60)
    print("\nNote: These tests verify code structure and imports.")
    print("They do NOT download/test the CLIP model itself.\n")
    
    tests = [
        ("Imports", test_imports),
        ("Model Fields", test_model_fields),
        ("Schema Fields", test_schema_fields),
        ("Configuration", test_config_values),
        ("CLI Script", test_cli_script),
        ("Training Script", test_training_script),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ All tests passed! Implementation structure is correct.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run examples: python examples/clip_examples.py")
        print("  3. Test with real images using the API")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
