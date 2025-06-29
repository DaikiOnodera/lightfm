#!/usr/bin/env python3
"""
Basic LightFM example without full numpy/scipy dependencies
Tests our PHP-compatible replacements
"""

from lightfm.datasets import fetch_movielens
from lightfm.lightfm import LightFM, predict_lightfm_pure

def main():
    print("Testing LightFM with PHP-compatible replacements...")
    
    # Test 1: Load data with our sparse matrix replacements
    print("1. Loading MovieLens data...")
    try:
        data = fetch_movielens()
        print(f"   ✅ Train shape: {data['train'].shape}")
        print(f"   ✅ Test shape: {data['test'].shape}")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return
    
    # Test 2: Create LightFM model with our replacements
    print("2. Creating LightFM model...")
    try:
        model = LightFM(no_components=5, random_state=42)
        print(f"   ✅ Model created with {model.no_components} components")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return
    
    # Test 3: Try basic model fitting (just 1 epoch to test)
    print("3. Testing model fitting...")
    try:
        import traceback
        model.fit_partial(data['train'], epochs=1)
        print("   ✅ Model fitting completed")
    except Exception as e:
        print(f"   ❌ Model fitting failed: {e}")
        print("   Traceback:")
        traceback.print_exc()
        return
        
    # Test 4: Try basic prediction with pure Python implementation
    print("4. Testing pure Python predictions...")
    try:
        # Use our pure Python prediction function
        predictions = predict_lightfm_pure(
            model.item_embeddings,
            model.user_embeddings,
            model.item_biases,
            model.user_biases,
            [0, 1], [0, 1]
        )
        print(f"   ✅ Pure Python predictions: {predictions}")
    except Exception as e:
        print(f"   ❌ Pure Python predictions failed: {e}")
        return
    
    print("\n🎉 All tests passed! LightFM works with PHP-compatible replacements!")

if __name__ == "__main__":
    main()
