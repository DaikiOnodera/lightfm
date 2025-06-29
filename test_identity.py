#!/usr/bin/env python3
"""
Test script to validate our custom identity function
works similarly to scipy's sp.identity.
"""

def test_identity_comparison():
    """Compare our identity function with scipy's sp.identity behavior"""
    
    # Import scipy for comparison (we'll remove this later)
    try:
        import scipy.sparse as sp
        scipy_available = True
    except ImportError:
        scipy_available = False
        print("Scipy not available, testing only custom implementation")
    
    # Define our custom implementation directly for testing
    class CsrMatrix:
        """Simple replacement for scipy.sparse.csr_matrix"""
        def __init__(self, shape, dtype=None):
            self.shape = shape
            self.dtype = dtype
            self.data = {}
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def __getitem__(self, key):
            return self.data.get(key, 0)

    def identity(n, format="csr", dtype=None):
        """Create an identity matrix - simple replacement for scipy.sparse.identity"""
        if format == "csr":
            mat = CsrMatrix((n, n), dtype=dtype)
            # Set diagonal elements to 1
            for i in range(n):
                mat[i, i] = 1.0
            return mat
        else:
            raise ValueError(f"Format '{format}' not supported")
    
    # Test parameters
    n = 5
    test_dtype = float
    
    # Test our custom identity function
    print("Testing custom identity function...")
    custom_mat = identity(n, format="csr", dtype=test_dtype)
    
    print(f"Custom matrix shape: {custom_mat.shape}")
    print(f"Custom matrix dtype: {custom_mat.dtype}")
    
    # Check diagonal elements
    diagonal_values = []
    for i in range(n):
        diagonal_values.append(custom_mat[i, i])
    print(f"Custom matrix diagonal: {diagonal_values}")
    
    # Check off-diagonal elements (should be 0)
    off_diagonal_values = []
    for i in range(min(2, n)):  # Check first 2 off-diagonal elements
        for j in range(min(2, n)):
            if i != j:
                off_diagonal_values.append(custom_mat[i, j])
    print(f"Custom matrix off-diagonal sample: {off_diagonal_values}")
    
    # Test scipy identity if available
    if scipy_available:
        print("\nTesting scipy sp.identity...")
        scipy_mat = sp.identity(n, format="csr", dtype=test_dtype)
        
        print(f"Scipy matrix shape: {scipy_mat.shape}")
        print(f"Scipy matrix dtype: {scipy_mat.dtype}")
        
        # Check diagonal elements
        scipy_diagonal = []
        for i in range(n):
            scipy_diagonal.append(scipy_mat[i, i])
        print(f"Scipy matrix diagonal: {scipy_diagonal}")
        
        # Check off-diagonal elements
        scipy_off_diagonal = []
        for i in range(min(2, n)):
            for j in range(min(2, n)):
                if i != j:
                    scipy_off_diagonal.append(scipy_mat[i, j])
        print(f"Scipy matrix off-diagonal sample: {scipy_off_diagonal}")
        
        # Compare results
        print("\nComparison:")
        shapes_match = custom_mat.shape == scipy_mat.shape
        dtypes_match = custom_mat.dtype == scipy_mat.dtype
        diagonal_match = diagonal_values == scipy_diagonal
        off_diagonal_match = off_diagonal_values == scipy_off_diagonal
        
        print(f"Shapes match: {shapes_match}")
        print(f"Dtypes match: {dtypes_match}")
        print(f"Diagonal values match: {diagonal_match}")
        print(f"Off-diagonal values match: {off_diagonal_match}")
        
        if all([shapes_match, dtypes_match, diagonal_match, off_diagonal_match]):
            print("\n✅ SUCCESS: Custom identity function behaves identically to scipy sp.identity!")
        else:
            print("\n❌ FAILURE: Custom identity function differs from scipy sp.identity")
            return False
    
    # Basic functionality test
    print("\nBasic functionality test:")
    assert custom_mat.shape == (n, n), f"Shape mismatch: expected ({n}, {n}), got {custom_mat.shape}"
    assert custom_mat.dtype == test_dtype, f"Dtype mismatch: expected {test_dtype}, got {custom_mat.dtype}"
    
    # Check that diagonal elements are 1
    for i in range(n):
        assert custom_mat[i, i] == 1.0, f"Diagonal element at ({i}, {i}) should be 1.0, got {custom_mat[i, i]}"
    
    # Check that off-diagonal elements are 0
    for i in range(n):
        for j in range(n):
            if i != j:
                assert custom_mat[i, j] == 0, f"Off-diagonal element at ({i}, {j}) should be 0, got {custom_mat[i, j]}"
    
    print("✅ Basic functionality test passed!")
    
    # Test different sizes
    print("\nTesting different matrix sizes:")
    for size in [1, 3, 10]:
        test_mat = identity(size, format="csr", dtype=float)
        assert test_mat.shape == (size, size)
        # Check a few diagonal elements
        for i in range(min(3, size)):
            assert test_mat[i, i] == 1.0
        print(f"✅ Size {size}x{size} test passed!")
    
    return True

if __name__ == "__main__":
    success = test_identity_comparison()
    if success:
        print("\n🎉 All identity function tests passed!")
    else:
        print("\n💥 Identity function tests failed!")
        exit(1)
