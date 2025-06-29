import sys
import os

# Test the helper functions we created in lightfm.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lightfm.lightfm import (
    create_identity_sparse, 
    create_csr_sparse, 
    create_zeros_like, 
    create_zeros,
    create_random_normal,
    RandomState,
    subtract_scalar_from_matrix,
    divide_matrix_by_scalar,
    np_arange,
    np_squeeze,
    np_less,
    np_any,
    np_all,
    np_logical_not,
    np_sort,
    np_concatenate,
    dot_product,
    matrix_vector_multiply,
    predict_lightfm_pure,
    sparse_matrix_multiply
)

def test_create_identity_sparse():
    """Test identity sparse matrix creation"""
    identity = create_identity_sparse(3)
    assert identity['shape'] == (3, 3)
    assert identity['type'] == 'identity'
    assert identity['data'][(0, 0)] == 1.0
    assert identity['data'][(1, 1)] == 1.0
    assert identity['data'][(2, 2)] == 1.0
    assert len(identity['data']) == 3

def test_create_csr_sparse():
    """Test CSR sparse matrix creation"""
    csr = create_csr_sparse((5, 3))
    assert csr['shape'] == (5, 3)
    assert csr['type'] == 'csr'
    assert len(csr['data']) == 0

def test_create_zeros():
    """Test zeros creation"""
    # 1D zeros
    zeros_1d = create_zeros(5)
    assert len(zeros_1d) == 5
    assert all(x == 0.0 for x in zeros_1d)
    
    # 2D zeros
    zeros_2d = create_zeros((3, 4))
    assert len(zeros_2d) == 3
    assert len(zeros_2d[0]) == 4
    assert all(all(x == 0.0 for x in row) for row in zeros_2d)

def test_create_zeros_like():
    """Test zeros_like creation"""
    # Template 1D
    template_1d = [1.0, 2.0, 3.0]
    zeros_1d = create_zeros_like(template_1d)
    assert len(zeros_1d) == 3
    assert all(x == 0.0 for x in zeros_1d)
    
    # Template 2D
    template_2d = [[1.0, 2.0], [3.0, 4.0]]
    zeros_2d = create_zeros_like(template_2d)
    assert len(zeros_2d) == 2
    assert len(zeros_2d[0]) == 2
    assert all(all(x == 0.0 for x in row) for row in zeros_2d)

def test_random_state():
    """Test RandomState functionality"""
    rs = RandomState(42)
    
    # Test rand method
    vals = rs.rand(5)
    assert len(vals) == 5
    assert all(0 <= x < 1 for x in vals)
    
    # Test 2D rand
    vals_2d = rs.rand(3, 2)
    assert len(vals_2d) == 3
    assert len(vals_2d[0]) == 2
    assert all(all(0 <= x < 1 for x in row) for row in vals_2d)

def test_matrix_operations():
    """Test matrix arithmetic operations"""
    # Test subtract scalar
    matrix_2d = [[1.0, 2.0], [3.0, 4.0]]
    result = subtract_scalar_from_matrix(matrix_2d, 0.5)
    assert result[0][0] == 0.5
    assert result[1][1] == 3.5
    
    # Test divide by scalar
    result2 = divide_matrix_by_scalar(result, 2.0)
    assert result2[0][0] == 0.25
    assert result2[1][1] == 1.75

def test_numpy_replacements():
    """Test numpy replacement functions"""
    
    # Test np_arange
    assert np_arange(5) == [0, 1, 2, 3, 4]
    assert np_arange(2, 5) == [2, 3, 4]
    assert np_arange(0, 10, 2) == [0, 2, 4, 6, 8]
    
    # Test np_squeeze
    assert np_squeeze([[[1, 2, 3]]]) == [1, 2, 3]
    assert np_squeeze([1, 2, 3]) == [1, 2, 3]
    
    # Test np_less
    assert np_less([1, 2, 3], [2, 2, 4]) == [True, False, True]
    assert np_less([1, 2, 3], 2) == [True, False, False]
    
    # Test np_any and np_all
    assert np_any([True, False, False]) == True
    assert np_any([False, False, False]) == False
    assert np_all([True, True, True]) == True
    assert np_all([True, False, True]) == False
    
    # Test np_logical_not
    assert np_logical_not([True, False, True]) == [False, True, False]
    
    # Test np_sort
    assert np_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
    
    # Test np_concatenate
    assert np_concatenate([[1, 2], [3, 4]]) == [1, 2, 3, 4]

def test_core_ml_algorithms():
    """Test core ML algorithm implementations"""
    
    # Test dot product
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [2.0, 1.0, 0.5]
    result = dot_product(vec1, vec2)
    expected = 1.0*2.0 + 2.0*1.0 + 3.0*0.5  # = 5.5
    assert abs(result - expected) < 1e-6
    
    # Test matrix vector multiply
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    vector = [1.0, 2.0]
    result = matrix_vector_multiply(matrix, vector)
    assert result == [5.0, 11.0]  # [1*1+2*2, 3*1+4*2]
    
    # Test sparse matrix multiply
    sparse_mat = {
        'data': {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0},
        'shape': (2, 2)
    }
    vector = [1.0, 2.0]
    result = sparse_matrix_multiply(sparse_mat, vector)
    assert result == [5.0, 3.0]  # [1*1+2*2, 3*1+0*2]
    
    # Test predict_lightfm_pure
    item_embeddings = [[1.0, 0.5], [0.5, 1.0]]
    user_embeddings = [[0.8, 0.2], [0.3, 0.7]]
    item_biases = [0.1, 0.2]
    user_biases = [0.05, 0.15]
    
    predictions = predict_lightfm_pure(
        item_embeddings, user_embeddings, item_biases, user_biases,
        [0, 1], [0, 1]  # user_ids, item_ids
    )
    
    # Should return 2 predictions
    assert len(predictions) == 2
    assert all(isinstance(p, float) for p in predictions)

def test_lightfm_initialization():
    """Test that LightFM can be imported and initialized with our changes"""
    from lightfm.lightfm import LightFM
    
    # Should not crash with our PHP-compatible replacements
    model = LightFM(no_components=5, random_state=42)
    assert model.no_components == 5
    assert model.random_state is not None

if __name__ == "__main__":
    test_create_identity_sparse()
    test_create_csr_sparse()
    test_create_zeros()
    test_create_zeros_like()
    test_random_state()
    test_matrix_operations()
    test_numpy_replacements()
    test_core_ml_algorithms()
    test_lightfm_initialization()
    print("✅ All LightFM helper function tests passed!")
