import itertools
import os
import zipfile

import numpy as np



from lightfm.datasets import _common


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
    
    def tocsr(self):
        """Convert to CSR format - returns a CsrMatrix"""
        csr_mat = CsrMatrix(self.shape, self.dtype)
        for (row, col), value in self.data.items():
            csr_mat[row, col] = value
        return csr_mat


class CooMatrix:
    """Simple replacement for scipy.sparse.coo_matrix"""
    def __init__(self, row, col, data, shape):
        self.row = row
        self.col = col
        self.data = data
        self.shape = shape
    
    def tocsr(self):
        """Convert to CSR format - returns a CsrMatrix"""
        csr_mat = CsrMatrix(self.shape, dtype=getattr(self, 'dtype', None))
        # Convert COO format to CSR format
        for i, (row, col) in enumerate(zip(self.row, self.col)):
            if i < len(self.data):
                csr_mat[row, col] = self.data[i]
        return csr_mat
    
    def getnnz(self, axis=None):
        """Get number of non-zero elements along given axis"""
        import numpy as np
        
        if axis is None:
            # Total number of non-zero elements
            return len(self.data)
        elif axis == 0:
            # Number of non-zeros per column
            col_nnz = {}
            for row, col in zip(self.row, self.col):
                col_nnz[col] = col_nnz.get(col, 0) + 1
            result = [col_nnz.get(i, 0) for i in range(self.shape[1])]
            return np.array(result)
        elif axis == 1:
            # Number of non-zeros per row
            row_nnz = {}
            for row, col in zip(self.row, self.col):
                row_nnz[row] = row_nnz.get(row, 0) + 1
            result = [row_nnz.get(i, 0) for i in range(self.shape[0])]
            return np.array(result)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D matrix")


class CsrMatrix:
    """Simple replacement for scipy.sparse.csr_matrix"""
    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype
        self._data_dict = {}  # Internal dict storage
        self._update_csr_arrays()
    
    def __setitem__(self, key, value):
        self._data_dict[key] = value
        self._update_csr_arrays()
    
    def __getitem__(self, key):
        return self._data_dict.get(key, 0)
    
    def _update_csr_arrays(self):
        """Update CSR format arrays (data, indices, indptr) from dict data"""
        import numpy as np
        
        if not self._data_dict:
            # Empty matrix
            self.data = np.array([], dtype=np.float32)  # For Cython compatibility
            self.indices = np.array([], dtype=np.int32)
            self.indptr = np.array([0] * (self.shape[0] + 1), dtype=np.int32)
            return
        
        # Sort by row, then by column
        sorted_items = sorted(self._data_dict.items(), key=lambda x: (x[0][0], x[0][1]))
        
        data_list = []
        indices_list = []
        indptr_list = [0]
        
        current_row = 0
        for (row, col), value in sorted_items:
            # Fill indptr for any missing rows
            while current_row < row:
                indptr_list.append(len(data_list))
                current_row += 1
            
            data_list.append(value)
            indices_list.append(col)
            current_row = row
        
        # Fill remaining indptr entries
        while current_row < self.shape[0]:
            indptr_list.append(len(data_list))
            current_row += 1
        
        # Convert to numpy arrays
        self.data = np.array(data_list, dtype=np.float32)  # For Cython compatibility
        self.indices = np.array(indices_list, dtype=np.int32)
        self.indptr = np.array(indptr_list, dtype=np.int32)
    
    @property
    def has_sorted_indices(self):
        """For compatibility with scipy - assume indices are always sorted"""
        return True
    
    def sorted_indices(self):
        """Return self since we assume indices are already sorted"""
        return self
    
    def astype(self, dtype):
        """Convert matrix to given dtype - for compatibility"""
        new_matrix = CsrMatrix(self.shape, dtype=dtype)
        new_matrix._data_dict = dict(self._data_dict)
        new_matrix._update_csr_arrays()
        return new_matrix
    
    def sum(self, axis=None):
        """Sum matrix elements along given axis"""
        if axis is None:
            # Sum all elements
            return sum(self._data_dict.values())
        elif axis == 0:
            # Sum along rows (return column sums)
            col_sums = {}
            for (row, col), value in self._data_dict.items():
                col_sums[col] = col_sums.get(col, 0) + value
            # Convert to list with zeros for missing columns
            result = [col_sums.get(i, 0) for i in range(self.shape[1])]
            return result
        elif axis == 1:
            # Sum along columns (return row sums)
            row_sums = {}
            for (row, col), value in self._data_dict.items():
                row_sums[row] = row_sums.get(row, 0) + value
            # Convert to list with zeros for missing rows
            result = [row_sums.get(i, 0) for i in range(self.shape[0])]
            return result
        else:
            raise ValueError(f"Invalid axis {axis} for 2D matrix")


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

    mat = LilMatrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, _ in data:
        if rating >= min_rating:
            mat[uid, iid] = rating

    return mat.tocoo()


def _parse_item_metadata(num_items, item_metadata_raw, genres_raw):

    genres = []

    for line in genres_raw:
        if line:
            genre, gid = line.split("|")
            genres.append("genre:{}".format(genre))

    id_feature_labels = np.empty(num_items, dtype=str)
    genre_feature_labels = np.array(genres)

    id_features = identity(num_items, format="csr", dtype=np.float32)
    genre_features = LilMatrix((num_items, len(genres)), dtype=np.float32)

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
            genre_features[iid, gid] = 1.0

    return (
        id_features,
        id_feature_labels,
        genre_features.tocsr(),
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

    assert train.shape == test.shape

    # Load metadata features
    (
        id_features,
        id_feature_labels,
        genre_features_matrix,
        genre_feature_labels,
    ) = _parse_item_metadata(num_items, item_metadata_raw, genres_raw)

    assert id_features.shape == (num_items, len(id_feature_labels))
    assert genre_features_matrix.shape == (num_items, len(genre_feature_labels))

    if indicator_features and not genre_features:
        features = id_features
        feature_labels = id_feature_labels
    elif genre_features and not indicator_features:
        features = genre_features_matrix
        feature_labels = genre_feature_labels
    else:
        features = hstack([id_features, genre_features_matrix])
        feature_labels = np.concatenate((id_feature_labels, genre_feature_labels))

    data = {
        "train": train,
        "test": test,
        "item_features": features,
        "item_feature_labels": feature_labels,
        "item_labels": id_feature_labels,
    }

    return data
