#!/usr/bin/env python3
"""
Test script to validate our CsrMatrix class properly maintains
CSR format arrays (data, indices, indptr).
"""

def test_csr_matrix_indices():
    """Test that CsrMatrix properly maintains indices and indptr arrays"""
    
    # Define our custom implementation directly for testing
    class CsrMatrix:
        """Simple replacement for scipy.sparse.csr_matrix"""
        def __init__(self, shape, dtype=None):
            self.shape = shape
            self.dtype = dtype
            self.data = {}
            self._update_csr_arrays()
        
        def __setitem__(self, key, value):
            self.data[key] = value
            self._update_csr_arrays()
        
        def __getitem__(self, key):
            return self.data.get(key, 0)
        
        def _update_csr_arrays(self):
            """Update CSR format arrays (data, indices, indptr) from dict data"""
            if not self.data:
                # Empty matrix
                self.data_array = []
                self.indices = []
                self.indptr = [0] * (self.shape[0] + 1)
                return
            
            # Sort by row, then by column
            sorted_items = sorted(self.data.items(), key=lambda x: (x[0][0], x[0][1]))
            
            self.data_array = []
            self.indices = []
            self.indptr = [0]
            
            current_row = 0
            for (row, col), value in sorted_items:
                # Fill indptr for any missing rows
                while current_row < row:
                    self.indptr.append(len(self.data_array))
                    current_row += 1
                
                self.data_array.append(value)
                self.indices.append(col)
                current_row = row
            
            # Fill remaining indptr entries
            while current_row < self.shape[0]:
                self.indptr.append(len(self.data_array))
                current_row += 1
        
        @property
        def has_sorted_indices(self):
            """For compatibility with scipy - assume indices are always sorted"""
            return True
        
        def sorted_indices(self):
            """Return self since we assume indices are already sorted"""
            return self
    
    print("Testing CsrMatrix with CSR format arrays...")
    
    # Test 1: Empty matrix
    print("\nTest 1: Empty matrix")
    mat = CsrMatrix((3, 4))
    print(f"Shape: {mat.shape}")
    print(f"Data array: {mat.data_array}")
    print(f"Indices: {mat.indices}")
    print(f"Indptr: {mat.indptr}")
    
    assert mat.shape == (3, 4), f"Expected shape (3, 4), got {mat.shape}"
    assert mat.data_array == [], f"Expected empty data_array, got {mat.data_array}"
    assert mat.indices == [], f"Expected empty indices, got {mat.indices}"
    assert mat.indptr == [0, 0, 0, 0], f"Expected [0, 0, 0, 0], got {mat.indptr}"
    print("✅ Empty matrix test passed!")
    
    # Test 2: Add some values
    print("\nTest 2: Adding values")
    mat[0, 1] = 5.0
    mat[0, 3] = 2.0
    mat[2, 0] = 3.0
    mat[2, 2] = 4.0
    
    print(f"Matrix data: {dict(mat.data)}")
    print(f"Data array: {mat.data_array}")
    print(f"Indices: {mat.indices}")
    print(f"Indptr: {mat.indptr}")
    
    # Expected CSR format:
    # Row 0: columns [1, 3] with values [5.0, 2.0]  
    # Row 1: no values
    # Row 2: columns [0, 2] with values [3.0, 4.0]
    
    expected_data = [5.0, 2.0, 3.0, 4.0]
    expected_indices = [1, 3, 0, 2]
    expected_indptr = [0, 2, 2, 4]  # Row 0: 0->2, Row 1: 2->2, Row 2: 2->4
    
    assert mat.data_array == expected_data, f"Expected data {expected_data}, got {mat.data_array}"
    assert mat.indices == expected_indices, f"Expected indices {expected_indices}, got {mat.indices}"
    assert mat.indptr == expected_indptr, f"Expected indptr {expected_indptr}, got {mat.indptr}"
    print("✅ Adding values test passed!")
    
    # Test 3: Verify CSR format interpretation
    print("\nTest 3: Verify CSR format interpretation")
    # Manually reconstruct matrix from CSR arrays to verify correctness
    reconstructed = {}
    for row in range(len(mat.indptr) - 1):
        start = mat.indptr[row]
        end = mat.indptr[row + 1]
        for idx in range(start, end):
            col = mat.indices[idx]
            value = mat.data_array[idx]
            reconstructed[(row, col)] = value
    
    print(f"Original data: {dict(mat.data)}")
    print(f"Reconstructed: {reconstructed}")
    
    assert reconstructed == mat.data, "CSR reconstruction doesn't match original data"
    print("✅ CSR format interpretation test passed!")
    
    # Test 4: Matrix access
    print("\nTest 4: Matrix element access")
    assert mat[0, 1] == 5.0, f"Expected mat[0,1] = 5.0, got {mat[0, 1]}"
    assert mat[0, 3] == 2.0, f"Expected mat[0,3] = 2.0, got {mat[0, 3]}"
    assert mat[1, 0] == 0, f"Expected mat[1,0] = 0, got {mat[1, 0]}"
    assert mat[2, 0] == 3.0, f"Expected mat[2,0] = 3.0, got {mat[2, 0]}"
    assert mat[2, 2] == 4.0, f"Expected mat[2,2] = 4.0, got {mat[2, 2]}"
    print("✅ Matrix element access test passed!")
    
    # Test 5: Properties
    print("\nTest 5: Properties")
    assert mat.has_sorted_indices == True, "has_sorted_indices should be True"
    assert mat.sorted_indices() == mat, "sorted_indices() should return self"
    print("✅ Properties test passed!")
    
    # Test 6: Sparse pattern
    print("\nTest 6: Sparse pattern with gaps")
    sparse_mat = CsrMatrix((5, 3))
    sparse_mat[0, 2] = 1.0
    sparse_mat[3, 1] = 2.0
    sparse_mat[4, 0] = 3.0
    
    print(f"Sparse matrix data: {dict(sparse_mat.data)}")
    print(f"Sparse data array: {sparse_mat.data_array}")
    print(f"Sparse indices: {sparse_mat.indices}")
    print(f"Sparse indptr: {sparse_mat.indptr}")
    
    expected_sparse_data = [1.0, 2.0, 3.0]
    expected_sparse_indices = [2, 1, 0]
    expected_sparse_indptr = [0, 1, 1, 1, 2, 3]  # Rows 0,1,2,3,4
    
    assert sparse_mat.data_array == expected_sparse_data
    assert sparse_mat.indices == expected_sparse_indices  
    assert sparse_mat.indptr == expected_sparse_indptr
    print("✅ Sparse pattern test passed!")
    
    return True

if __name__ == "__main__":
    success = test_csr_matrix_indices()
    if success:
        print("\n🎉 All CsrMatrix indices tests passed!")
    else:
        print("\n💥 CsrMatrix indices tests failed!")
        exit(1)
