import numpy as np
import scipy.sparse as sp

from lightfm.lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.datasets.movielens import create_identity_sparse, create_lil_sparse, set_sparse_value, hstack_sparse

SEED = 10

def test_create_identity_sparse():
    """Test create_identity_sparse function"""
    size = 3
    identity = create_identity_sparse(size)
    assert identity['shape'] == (3, 3)
    assert identity['type'] == 'identity'
    assert identity['data'][(0, 0)] == 1.0
    assert len(identity['data']) == 3

def test_create_lil_sparse():
    """Test create_lil_sparse function"""
    shape = (3, 4)
    matrix = create_lil_sparse(shape)
    assert matrix['shape'] == (3, 4)
    assert matrix['type'] == 'lil'
    assert len(matrix['data']) == 0

def test_hstack_sparse():
    """Test hstack_sparse function"""
    mat1 = create_lil_sparse((3, 2))
    set_sparse_value(mat1, 0, 0, 1.0)
    
    mat2 = create_lil_sparse((3, 3))
    set_sparse_value(mat2, 0, 1, 3.0)
    
    result = hstack_sparse([mat1, mat2])
    assert result['shape'] == (3, 5)
    assert result['data'][(0, 0)] == 1.0
    assert result['data'][(0, 3)] == 3.0

def test_movielens_basic():
    """Test basic MovieLens functionality"""
    data = fetch_movielens()
    
    # Check that we get our custom sparse matrices
    assert hasattr(data["train"], '_dict')
    assert hasattr(data["test"], '_dict')
    
    # Check basic properties work
    assert data["train"].shape[0] > 0
    assert data["test"].shape[1] > 0
    
    # Test basic model fitting
    model = LightFM(random_state=SEED)
    model.fit_partial(data["train"], epochs=1)
    
    # Test predictions work
    train_predictions = model.predict([0, 1], [0, 1])
    assert len(train_predictions) == 2

if __name__ == "__main__":
    test_create_identity_sparse()
    test_create_lil_sparse() 
    test_hstack_sparse()
    test_movielens_basic()
    print("✅ All core functionality tests passed!")
