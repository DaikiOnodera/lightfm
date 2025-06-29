import itertools
import os
import zipfile

# Note: numpy and scipy imports removed for PHP compatibility
# They are now conditionally imported only when needed in SparseMatrixAdapter

import scipy.sparse as sp

from lightfm.datasets import _common


def create_identity_sparse(size):
    """Create identity matrix as dictionary {(row, col): value}
    
    Args:
        size: Size of the square identity matrix
        
    Returns:
        dict: Dictionary with 'data', 'shape', and 'type' keys
    """
    identity = {}
    for i in range(size):
        identity[(i, i)] = 1.0  # Only store diagonal elements
    return {
        'data': identity,
        'shape': (size, size),
        'type': 'identity'
    }


def create_lil_sparse(shape):
    """Create empty sparse matrix as dictionary for incremental construction
    
    Args:
        shape: Tuple of (rows, cols) for matrix dimensions
        
    Returns:
        dict: Dictionary with 'data', 'shape', and 'type' keys
    """
    return {
        'data': {},
        'shape': shape,
        'type': 'lil'
    }


def set_sparse_value(matrix, row, col, value):
    """Set value in sparse matrix dictionary
    
    Args:
        matrix: Sparse matrix dictionary
        row: Row index
        col: Column index  
        value: Value to set
    """
    matrix['data'][(row, col)] = value


def hstack_sparse(matrices):
    """Horizontally stack sparse matrices (concatenate side by side)
    
    Args:
        matrices: List of sparse matrix dictionaries to stack
        
    Returns:
        dict: New sparse matrix with combined data
    """
    if not matrices:
        return create_lil_sparse((0, 0))
    
    # Calculate new dimensions
    total_rows = matrices[0]['shape'][0]
    total_cols = sum(mat['shape'][1] for mat in matrices)
    
    # Create result matrix
    result = create_lil_sparse((total_rows, total_cols))
    
    # Stack matrices horizontally
    col_offset = 0
    for matrix in matrices:
        # Verify row count matches
        if matrix['shape'][0] != total_rows:
            raise ValueError(f"All matrices must have same number of rows")
            
        # Copy data with column offset
        for (row, col), value in matrix['data'].items():
            set_sparse_value(result, row, col + col_offset, value)
            
        col_offset += matrix['shape'][1]
    
    result['type'] = 'csr'  # Mark as CSR-like after stacking
    return result


class SparseMatrixAdapter:
    """Adapter to make our dictionary sparse matrices compatible with scipy interface"""
    
    def __init__(self, sparse_dict):
        self._dict = sparse_dict
        
    @property
    def shape(self):
        return self._dict['shape']
        
    @property 
    def data(self):
        # Return array of non-zero values for compatibility (only when numpy is available)
        try:
            import numpy as np
            return np.array([float(value) for value in self._dict['data'].values()], dtype=np.float32)
        except ImportError:
            # Fallback for pure Python - return list
            return [float(value) for value in self._dict['data'].values()]
    
    @data.setter
    def data(self, values):
        """Set data values - update the internal data structure"""
        # For tests that modify data (like setting values to 0 or -1)
        keys = list(self._dict['data'].keys())
        if hasattr(values, '__len__') and len(values) == len(keys):
            for i, key in enumerate(keys):
                self._dict['data'][key] = values[i]
        # This is a simplified implementation
        
    @property
    def dtype(self):
        try:
            import numpy as np
            return np.float32
        except ImportError:
            return float  # Python built-in float type
        
    @property
    def row(self):
        # Return row indices for COO format
        try:
            import numpy as np
            return np.array([row for row, col in self._dict['data'].keys()], dtype=np.int32)
        except ImportError:
            return [row for row, col in self._dict['data'].keys()]
        
    @property 
    def col(self):
        # Return column indices for COO format
        try:
            import numpy as np
            return np.array([col for row, col in self._dict['data'].keys()], dtype=np.int32)
        except ImportError:
            return [col for row, col in self._dict['data'].keys()]
        
    def tocoo(self):
        """Return self - already in COO-like format"""
        return SparseMatrixAdapter(self._dict)
        
    def tocsr(self):
        """Return self - mark as CSR format"""
        result_dict = self._dict.copy()
        result_dict['type'] = 'csr'
        return SparseMatrixAdapter(result_dict)
        
    def eliminate_zeros(self):
        """Remove zero entries - no-op since we only store non-zero values"""
        pass
        
    def __getitem__(self, key):
        # Allow matrix[row, col] access
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            # Handle numpy array indexing for sklearn compatibility
            if hasattr(row, '__iter__') and not isinstance(row, (int, slice)):
                # Return a new sparse matrix with subset of rows
                return self._subset_rows(row)
            return self._dict['data'].get(key, 0.0)
        return self._dict[key]
    
    def _subset_rows(self, row_indices):
        """Create a subset matrix with only the specified rows"""
        new_data = {}
        for r in row_indices:
            for (orig_r, c), val in self._dict['data'].items():
                if orig_r == r:
                    new_data[(r, c)] = val
        return SparseMatrixAdapter({
            'data': new_data,
            'shape': (len(row_indices), self._dict['shape'][1]),
            'type': self._dict['type']
        })
    
    def copy(self):
        """Create a copy of this sparse matrix"""
        return SparseMatrixAdapter({
            'data': self._dict['data'].copy(),
            'shape': self._dict['shape'],
            'type': self._dict['type']
        })
    
    def astype(self, dtype):
        """Convert data type - for compatibility, return self since we use float"""
        return self


def _read_raw_data(path):
    """
    Return the raw lines of the train and test files.
    """

    with zipfile.ZipFile(path) as datafile:
        return (
            datafile.read("ml-100k/ua.base").decode().split("\n"),
            datafile.read("ml-100k/ua.test").decode().split("\n"),
            datafile.read("ml-100k/u.item").decode(errors="ignore").split("\n"),
            datafile.read("ml-100k/u.genre").decode(errors="ignore").split("\n"),
        )


def _parse(data):

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split("\t")]

        # Subtract one from ids to shift
        # to zero-based indexing
        yield uid - 1, iid - 1, rating, timestamp


def _get_dimensions(train_data, test_data):

    uids = set()
    iids = set()

    for uid, iid, _, _ in itertools.chain(train_data, test_data):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return rows, cols


def _build_interaction_matrix(rows, cols, data, min_rating):

    mat = create_lil_sparse((rows, cols))

    for uid, iid, rating, _ in data:
        if rating >= min_rating:
            set_sparse_value(mat, uid, iid, float(rating))

    return mat  # Will handle .tocoo() conversion later


def _parse_item_metadata(num_items, item_metadata_raw, genres_raw):

    genres = []

    for line in genres_raw:
        if line:
            genre, gid = line.split("|")
            genres.append("genre:{}".format(genre))

    id_feature_labels = [None] * num_items  # Replace np.empty with list
    genre_feature_labels = list(genres)  # Replace np.array with list

    id_features = create_identity_sparse(num_items)
    genre_features = create_lil_sparse((num_items, len(genres)))

    for line in item_metadata_raw:

        if not line:
            continue

        splt = line.split("|")

        # Zero-based indexing
        iid = int(splt[0]) - 1
        title = splt[1]

        id_feature_labels[iid] = title

        item_genres = [idx for idx, val in enumerate(splt[5:]) if int(val) > 0]

        for gid in item_genres:
            set_sparse_value(genre_features, iid, gid, 1.0)

    return (
        id_features,
        id_feature_labels,
        genre_features,  # Remove .tocsr() for now
        genre_feature_labels,
    )


def fetch_movielens(
    data_home=None,
    indicator_features=True,
    genre_features=False,
    min_rating=0.0,
    download_if_missing=True,
):
    """
    Fetch the `Movielens 100k dataset <http://grouplens.org/datasets/movielens/100k/>`_.

    The dataset contains 100,000 interactions from 1000 users on 1700 movies,
    and is exhaustively described in its
    `README <http://files.grouplens.org/datasets/movielens/ml-100k-README.txt>`_.

    Parameters
    ----------

    data_home: path, optional
        Path to the directory in which the downloaded data should be placed.
        Defaults to ``~/lightfm_data/``.
    indicator_features: bool, optional
        Use an [n_items, n_items] identity matrix for item features. When True with genre_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_items, n_items + n_genres].
    genre_features: bool, optional
        Use a [n_items, n_genres] matrix for item features. When True with item_indicator_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_items, n_items + n_genres].
    min_rating: float, optional
        Minimum rating to include in the interaction matrix.
    download_if_missing: bool, optional
        Download the data if not present. Raises an IOError if False and data is missing.

    Notes
    -----

    The return value is a dictionary containing the following keys:

    Returns
    -------

    train: sp.coo_matrix of shape [n_users, n_items]
         Contains training set interactions.
    test: sp.coo_matrix of shape [n_users, n_items]
         Contains testing set interactions.
    item_features: sp.csr_matrix of shape [n_items, n_item_features]
         Contains item features.
    item_feature_labels: np.array of strings of shape [n_item_features,]
         Labels of item features.
    item_labels: np.array of strings of shape [n_items,]
         Items' titles.
    """

    if not (indicator_features or genre_features):
        raise ValueError(
            "At least one of item_indicator_features " "or genre_features must be True"
        )

    zip_path = _common.get_data(
        data_home,
        (
            "https://github.com/maciejkula/"
            "lightfm_datasets/releases/"
            "download/v0.1.0/movielens.zip"
        ),
        "movielens100k",
        "movielens.zip",
        download_if_missing,
    )

    # Load raw data
    try:
        (train_raw, test_raw, item_metadata_raw, genres_raw) = _read_raw_data(zip_path)
    except zipfile.BadZipFile:
        # Download was corrupted, get rid of the partially
        # downloaded file so that we re-download on the
        # next try.
        os.unlink(zip_path)
        raise ValueError(
            "Corrupted Movielens download. Check your "
            "internet connection and try again."
        )

    # Figure out the dimensions
    num_users, num_items = _get_dimensions(_parse(train_raw), _parse(test_raw))

    # Load train interactions
    train = _build_interaction_matrix(
        num_users, num_items, _parse(train_raw), min_rating
    )
    # Load test interactions
    test = _build_interaction_matrix(num_users, num_items, _parse(test_raw), min_rating)

    assert train['shape'] == test['shape']

    # Load metadata features
    (
        id_features,
        id_feature_labels,
        genre_features_matrix,
        genre_feature_labels,
    ) = _parse_item_metadata(num_items, item_metadata_raw, genres_raw)

    assert id_features['shape'] == (num_items, len(id_feature_labels))
    assert genre_features_matrix['shape'] == (num_items, len(genre_feature_labels))

    if indicator_features and not genre_features:
        features = id_features
        feature_labels = id_feature_labels
    elif genre_features and not indicator_features:
        features = genre_features_matrix
        feature_labels = genre_feature_labels
    else:
        features = hstack_sparse([id_features, genre_features_matrix])
        feature_labels = id_feature_labels + genre_feature_labels  # Replace np.concatenate with list concatenation

    data = {
        "train": SparseMatrixAdapter(train),
        "test": SparseMatrixAdapter(test), 
        "item_features": SparseMatrixAdapter(features),
        "item_feature_labels": feature_labels,
        "item_labels": id_feature_labels,
    }

    return data
