#!/usr/bin/env python3
"""
Test script to validate our custom hstack function
works similarly to scipy's sp.hstack.
"""

def test_hstack_comparison():
    """Compare our hstack function with scipy's sp.hstack behavior"""
    
    # Import scipy for comparison (we'll remove this later)
    try:
        import scipy.sparse as sp
        import numpy as np
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

    class LilMatrix:
        """Simple replacement for scipy.sparse.lil_matrix"""
        def __init__(self, shape, dtype=None):
            self.shape = shape
            self.dtype = dtype
            self.data = {}
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def __getitem__(self, key):
            return self.data.get(key, 0)
        
        def tocsr(self):
            """Convert to CSR format - returns a CsrMatrix"""
            csr_mat = CsrMatrix(self.shape, self.dtype)
            for (row, col), value in self.data.items():
                csr_mat[row, col] = value
            return csr_mat

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

    class HStackMatrix:
        """Simple matrix that horizontally stacks two matrices"""
        def __init__(self, left_matrix, right_matrix):
            self.left_matrix = left_matrix
            self.right_matrix = right_matrix
            
            # Ensure both matrices have same number of rows
            if left_matrix.shape[0] != right_matrix.shape[0]:
                raise ValueError("Matrices must have same number of rows for horizontal stacking")
            
            # Combined shape: same rows, combined columns
            self.shape = (left_matrix.shape[0], left_matrix.shape[1] + right_matrix.shape[1])
            self.left_cols = left_matrix.shape[1]
        
        def __getitem__(self, key):
            row, col = key
            if col < self.left_cols:
                # Access left matrix
                return self.left_matrix[row, col]
            else:
                # Access right matrix, adjust column index
                return self.right_matrix[row, col - self.left_cols]

    def hstack(matrices):
        """Horizontally stack matrices - simple replacement for scipy.sparse.hstack"""
        if len(matrices) != 2:
            raise ValueError("Currently only supports stacking 2 matrices")
        
        left, right = matrices
        return HStackMatrix(left, right)
    
    # Test parameters - create two simple matrices to stack
    rows = 3
    left_cols = 2
    right_cols = 3
    
    print("Testing custom hstack function...")
    
    # Create left matrix (3x2 identity-like)
    left_custom = CsrMatrix((rows, left_cols), dtype=float)
    left_custom[0, 0] = 1.0
    left_custom[1, 1] = 1.0
    left_custom[2, 0] = 0.5
    
    # Create right matrix (3x3 with some values)
    right_custom = LilMatrix((rows, right_cols), dtype=float)
    right_custom[0, 0] = 2.0
    right_custom[1, 2] = 3.0
    right_custom[2, 1] = 4.0
    
    # Stack them with our custom function
    custom_stacked = hstack([left_custom, right_custom])
    
    print(f"Custom stacked matrix shape: {custom_stacked.shape}")
    print(f"Expected shape: ({rows}, {left_cols + right_cols})")
    
    # Test accessing elements from both left and right sides
    print("\nTesting element access:")
    test_positions = [
        (0, 0),  # Left matrix
        (1, 1),  # Left matrix
        (2, 0),  # Left matrix
        (0, 2),  # Right matrix (col 0 in right = col 2 in stacked)
        (1, 4),  # Right matrix (col 2 in right = col 4 in stacked)
        (2, 3),  # Right matrix (col 1 in right = col 3 in stacked)
    ]
    
    custom_values = []
    for row, col in test_positions:
        value = custom_stacked[row, col]
        custom_values.append(value)
        print(f"Custom matrix[{row}, {col}] = {value}")
    
    # Test scipy hstack if available
    if scipy_available:
        print("\nTesting scipy sp.hstack...")
        
        # Create equivalent scipy matrices
        left_scipy = sp.csr_matrix((rows, left_cols), dtype=float)
        left_scipy[0, 0] = 1.0
        left_scipy[1, 1] = 1.0
        left_scipy[2, 0] = 0.5
        
        right_scipy = sp.lil_matrix((rows, right_cols), dtype=float)
        right_scipy[0, 0] = 2.0
        right_scipy[1, 2] = 3.0
        right_scipy[2, 1] = 4.0
        
        # Stack them with scipy
        scipy_stacked = sp.hstack([left_scipy, right_scipy]).tocsr()  # Convert to CSR for element access
        
        print(f"Scipy stacked matrix shape: {scipy_stacked.shape}")
        
        # Test accessing the same elements
        scipy_values = []
        for row, col in test_positions:
            value = scipy_stacked[row, col]
            scipy_values.append(value)
            print(f"Scipy matrix[{row}, {col}] = {value}")
        
        # Compare results
        print("\nComparison:")
        shapes_match = custom_stacked.shape == scipy_stacked.shape
        values_match = custom_values == scipy_values
        
        print(f"Shapes match: {shapes_match}")
        print(f"Values match: {values_match}")
        print(f"Custom values: {custom_values}")
        print(f"Scipy values: {scipy_values}")
        
        if shapes_match and values_match:
            print("\n✅ SUCCESS: Custom hstack function behaves identically to scipy sp.hstack!")
        else:
            print("\n❌ FAILURE: Custom hstack function differs from scipy sp.hstack")
            return False
    
    # Basic functionality tests
    print("\nBasic functionality tests:")
    
    # Test shape
    expected_shape = (rows, left_cols + right_cols)
    assert custom_stacked.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {custom_stacked.shape}"
    
    # Test left side access (should match original left matrix)
    assert custom_stacked[0, 0] == 1.0, f"Left matrix access failed: expected 1.0, got {custom_stacked[0, 0]}"
    assert custom_stacked[1, 1] == 1.0, f"Left matrix access failed: expected 1.0, got {custom_stacked[1, 1]}"
    assert custom_stacked[2, 0] == 0.5, f"Left matrix access failed: expected 0.5, got {custom_stacked[2, 0]}"
    
    # Test right side access (should match original right matrix)
    assert custom_stacked[0, 2] == 2.0, f"Right matrix access failed: expected 2.0, got {custom_stacked[0, 2]}"
    assert custom_stacked[1, 4] == 3.0, f"Right matrix access failed: expected 3.0, got {custom_stacked[1, 4]}"
    assert custom_stacked[2, 3] == 4.0, f"Right matrix access failed: expected 4.0, got {custom_stacked[2, 3]}"
    
    # Test zero elements
    assert custom_stacked[0, 1] == 0, f"Zero element access failed: expected 0, got {custom_stacked[0, 1]}"
    assert custom_stacked[1, 0] == 0, f"Zero element access failed: expected 0, got {custom_stacked[1, 0]}"
    
    print("✅ All basic functionality tests passed!")
    
    # Test error handling
    print("\nTesting error handling:")
    try:
        # Try to stack matrices with different row counts
        bad_left = CsrMatrix((2, 2), dtype=float)  # 2 rows
        bad_right = CsrMatrix((3, 2), dtype=float)  # 3 rows
        hstack([bad_left, bad_right])
        print("❌ Should have raised an error for mismatched row counts")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error for mismatched rows: {e}")
    
    try:
        # Try to stack more than 2 matrices
        hstack([left_custom, right_custom, left_custom])
        print("❌ Should have raised an error for more than 2 matrices")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error for too many matrices: {e}")
    
    return True

if __name__ == "__main__":
    success = test_hstack_comparison()
    if success:
        print("\n🎉 All hstack function tests passed!")
    else:
        print("\n💥 Hstack function tests failed!")
        exit(1)
