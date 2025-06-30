#!/usr/bin/env python3
"""
Test script to validate our custom LilMatrix implementation
works similarly to scipy's lil_matrix.
"""

def test_lil_matrix_comparison():
    """Compare our LilMatrix with scipy's lil_matrix behavior"""
    
    # Import scipy for comparison (we'll remove this later)
    try:
        import scipy.sparse as sp
        scipy_available = True
    except ImportError:
        scipy_available = False
        print("Scipy not available, testing only custom implementation")
    
    # Define our custom implementation directly for testing
    class LilMatrix:
        """Simple replacement for scipy.sparse.lil_matrix"""
        def __init__(self, shape, dtype=None):
            self.shape = shape
            self.dtype = dtype
            self.data = {}
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def tocoo(self):
            """Convert to COO format - returns a simple object with row, col, data arrays"""
            rows = []
            cols = []
            data = []
            
            for (row, col), value in self.data.items():
                rows.append(row)
                cols.append(col)
                data.append(value)
            
            return CooMatrix(rows, cols, data, self.shape)

    class CooMatrix:
        """Simple replacement for scipy.sparse.coo_matrix"""
        def __init__(self, row, col, data, shape):
            self.row = row
            self.col = col
            self.data = data
            self.shape = shape
    
    # Test parameters
    shape = (5, 4)
    test_data = [(0, 1, 3), (1, 2, 4), (2, 0, 5), (3, 3, 2)]
    
    # Test our custom LilMatrix
    print("Testing custom LilMatrix...")
    custom_mat = LilMatrix(shape)
    
    for row, col, value in test_data:
        custom_mat[row, col] = value
    
    custom_coo = custom_mat.tocoo()
    
    print(f"Custom matrix shape: {custom_coo.shape}")
    print(f"Custom matrix data: {custom_coo.data}")
    print(f"Custom matrix rows: {custom_coo.row}")
    print(f"Custom matrix cols: {custom_coo.col}")
    
    # Test scipy lil_matrix if available
    if scipy_available:
        print("\nTesting scipy lil_matrix...")
        scipy_mat = sp.lil_matrix(shape, dtype=int)
        
        for row, col, value in test_data:
            scipy_mat[row, col] = value
        
        scipy_coo = scipy_mat.tocoo()
        
        print(f"Scipy matrix shape: {scipy_coo.shape}")
        print(f"Scipy matrix data: {scipy_coo.data.tolist()}")
        print(f"Scipy matrix rows: {scipy_coo.row.tolist()}")
        print(f"Scipy matrix cols: {scipy_coo.col.tolist()}")
        
        # Compare results
        print("\nComparison:")
        shapes_match = custom_coo.shape == scipy_coo.shape
        data_match = custom_coo.data == scipy_coo.data.tolist()
        rows_match = custom_coo.row == scipy_coo.row.tolist()
        cols_match = custom_coo.col == scipy_coo.col.tolist()
        
        print(f"Shapes match: {shapes_match}")
        print(f"Data match: {data_match}")
        print(f"Rows match: {rows_match}")
        print(f"Cols match: {cols_match}")
        
        if all([shapes_match, data_match, rows_match, cols_match]):
            print("\n✅ SUCCESS: Custom LilMatrix behaves identically to scipy lil_matrix!")
        else:
            print("\n❌ FAILURE: Custom LilMatrix differs from scipy lil_matrix")
            return False
    
    # Basic functionality test
    print("\nBasic functionality test:")
    assert custom_coo.shape == shape, f"Shape mismatch: expected {shape}, got {custom_coo.shape}"
    assert len(custom_coo.data) == len(test_data), f"Data length mismatch"
    
    print("✅ Basic functionality test passed!")
    return True

if __name__ == "__main__":
    success = test_lil_matrix_comparison()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        exit(1)
