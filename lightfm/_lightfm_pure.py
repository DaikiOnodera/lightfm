#!/usr/bin/env python3
"""
Pure Python replacement for _lightfm_fast.pyx.template
Converts Cython code to use our custom CsrMatrix without numpy dependencies.
"""

import random
import math


class CSRMatrix:
    """
    Pure Python utility class for accessing elements of a CSR matrix.
    Replacement for the Cython CSRMatrix class.
    """
    
    def __init__(self, csr_matrix):
        # Use our custom CsrMatrix attributes
        self.indices = csr_matrix.indices if hasattr(csr_matrix, 'indices') else []
        self.indptr = csr_matrix.indptr if hasattr(csr_matrix, 'indptr') else []
        self.data = csr_matrix.data if hasattr(csr_matrix, 'data') else []
        
        self.rows, self.cols = csr_matrix.shape
        self.nnz = len(self.data)
    
    def get_row_start(self, row):
        """Return the pointer to the start of the data for row."""
        return self.indptr[row] if row < len(self.indptr) else 0
    
    def get_row_end(self, row):
        """Return the pointer to the end of the data for row."""
        return self.indptr[row + 1] if row + 1 < len(self.indptr) else len(self.data)


class FastLightFM:
    """
    Pure Python class holding all the model state.
    Replacement for the Cython FastLightFM class.
    """
    
    def __init__(self, item_embeddings, item_embedding_gradients, item_embedding_momentum,
                 item_biases, item_bias_gradients, item_bias_momentum,
                 user_embeddings, user_embedding_gradients, user_embedding_momentum,
                 user_biases, user_bias_gradients, user_bias_momentum,
                 no_components, adadelta, learning_rate, rho, epsilon, max_sampled):
        
        # Store all the model parameters
        self.item_embeddings = item_embeddings
        self.item_embedding_gradients = item_embedding_gradients
        self.item_embedding_momentum = item_embedding_momentum
        
        self.item_biases = item_biases
        self.item_bias_gradients = item_bias_gradients
        self.item_bias_momentum = item_bias_momentum
        
        self.user_embeddings = user_embeddings
        self.user_embedding_gradients = user_embedding_gradients
        self.user_embedding_momentum = user_embedding_momentum
        
        self.user_biases = user_biases
        self.user_bias_gradients = user_bias_gradients
        self.user_bias_momentum = user_bias_momentum
        
        # Store hyperparameters
        self.no_components = no_components
        self.adadelta = adadelta
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.max_sampled = max_sampled


def sigmoid(x):
    """Safe sigmoid function"""
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-x))


def dot_product(vec1, vec2):
    """Compute dot product of two vectors (lists or dicts)"""
    if isinstance(vec1, dict) and isinstance(vec2, dict):
        result = 0.0
        for key in vec1:
            if key in vec2:
                result += vec1[key] * vec2[key]
        return result
    elif hasattr(vec1, '__len__') and hasattr(vec2, '__len__'):
        result = 0.0
        for i in range(min(len(vec1), len(vec2))):
            result += vec1[i] * vec2[i]
        return result
    else:
        return 0.0


def compute_prediction(item_features, item_biases, user_features, user_biases,
                      item_id, user_id):
    """Compute prediction for user-item pair"""
    prediction = 0.0
    
    # Add biases
    if hasattr(item_biases, '__getitem__'):
        prediction += item_biases[item_id] if item_id < len(item_biases) else 0.0
    if hasattr(user_biases, '__getitem__'):
        prediction += user_biases[user_id] if user_id < len(user_biases) else 0.0
    
    # Add feature dot product
    if hasattr(item_features, '__getitem__') and hasattr(user_features, '__getitem__'):
        if item_id < len(item_features) and user_id < len(user_features):
            prediction += dot_product(item_features[item_id], user_features[user_id])
    
    return prediction


def fit_logistic(item_features, user_features, lightfm, interactions, sample_weight,
                shuffle_indices, lightfm_data, learning_rate, item_alpha, user_alpha, num_threads):
    """
    Pure Python implementation of logistic loss fitting.
    Simplified version of the Cython fit_logistic function.
    """
    print("Warning: Using simplified pure Python fit_logistic implementation")
    
    # Basic logistic regression training loop
    for idx in shuffle_indices:
        if idx >= len(interactions.data):
            continue
            
        # Get user-item pair
        user_id = interactions.row[idx] if idx < len(interactions.row) else 0
        item_id = interactions.col[idx] if idx < len(interactions.col) else 0
        rating = interactions.data[idx] if idx < len(interactions.data) else 0.0
        
        # Compute prediction
        prediction = compute_prediction(
            item_features, lightfm.item_biases,
            user_features, lightfm.user_biases,
            item_id, user_id
        )
        
        # Compute loss and gradients (simplified)
        predicted_prob = sigmoid(prediction)
        error = rating - predicted_prob
        
        # Update biases (simplified)
        if hasattr(lightfm.item_biases, '__setitem__') and item_id < len(lightfm.item_biases):
            lightfm.item_biases[item_id] += learning_rate * error
        if hasattr(lightfm.user_biases, '__setitem__') and user_id < len(lightfm.user_biases):
            lightfm.user_biases[user_id] += learning_rate * error


def fit_warp(item_features, user_features, positives_lookup, interactions_row, interactions_col, 
            interactions_data, sample_weight, shuffle_indices, lightfm_data, learning_rate, 
            item_alpha, user_alpha, num_threads, random_state):
    """
    Pure Python implementation of WARP loss fitting.
    Simplified version of the Cython fit_warp function.
    """
    print("Warning: Using simplified pure Python fit_warp implementation")
    
    # Basic WARP training loop (simplified)
    for idx in shuffle_indices:
        if idx >= len(interactions_data):
            continue
            
        # Get positive user-item pair
        user_id = interactions_row[idx] if idx < len(interactions_row) else 0
        item_id = interactions_col[idx] if idx < len(interactions_col) else 0
        
        # Compute positive prediction
        pos_prediction = compute_prediction(
            item_features, lightfm_data.item_biases,
            user_features, lightfm_data.user_biases,
            item_id, user_id
        )
        
        # Sample negative item (simplified)
        neg_item_id = random.randint(0, max(1, len(lightfm_data.item_biases) - 1))
        
        # Compute negative prediction
        neg_prediction = compute_prediction(
            item_features, lightfm_data.item_biases,
            user_features, lightfm_data.user_biases,
            neg_item_id, user_id
        )
        
        # WARP loss update (simplified)
        margin = 1.0
        if pos_prediction - neg_prediction < margin:
            # Update parameters
            if hasattr(lightfm_data.item_biases, '__setitem__'):
                if item_id < len(lightfm_data.item_biases):
                    lightfm_data.item_biases[item_id] += learning_rate
                if neg_item_id < len(lightfm_data.item_biases):
                    lightfm_data.item_biases[neg_item_id] -= learning_rate


def fit_warp_kos(item_features, user_features, lightfm, interactions, sample_weight,
                shuffle_indices, lightfm_data, learning_rate, item_alpha, user_alpha, num_threads):
    """
    Pure Python implementation of WARP-KOS loss fitting.
    Simplified version of the Cython fit_warp_kos function.
    """
    print("Warning: Using simplified pure Python fit_warp_kos implementation")
    # For now, just call regular WARP
    return fit_warp(item_features, user_features, None, lightfm, interactions, sample_weight,
                   shuffle_indices, lightfm_data, learning_rate, item_alpha, user_alpha, num_threads)


def fit_bpr(item_features, user_features, lightfm, interactions, sample_weight,
           shuffle_indices, lightfm_data, learning_rate, item_alpha, user_alpha, num_threads):
    """
    Pure Python implementation of BPR loss fitting.
    Simplified version of the Cython fit_bpr function.
    """
    print("Warning: Using simplified pure Python fit_bpr implementation")
    
    # Basic BPR training loop (simplified)
    for idx in shuffle_indices:
        if idx >= len(interactions.data):
            continue
            
        # Get positive user-item pair
        user_id = interactions.row[idx] if idx < len(interactions.row) else 0
        item_id = interactions.col[idx] if idx < len(interactions.col) else 0
        
        # Sample negative item
        neg_item_id = random.randint(0, max(1, len(lightfm.item_biases) - 1))
        
        # Compute predictions
        pos_prediction = compute_prediction(
            item_features, lightfm.item_biases,
            user_features, lightfm.user_biases,
            item_id, user_id
        )
        
        neg_prediction = compute_prediction(
            item_features, lightfm.item_biases,
            user_features, lightfm.user_biases,
            neg_item_id, user_id
        )
        
        # BPR loss update (simplified)
        diff = pos_prediction - neg_prediction
        sigmoid_diff = sigmoid(diff)
        
        # Update biases
        gradient = 1.0 - sigmoid_diff
        if hasattr(lightfm.item_biases, '__setitem__'):
            if item_id < len(lightfm.item_biases):
                lightfm.item_biases[item_id] += learning_rate * gradient
            if neg_item_id < len(lightfm.item_biases):
                lightfm.item_biases[neg_item_id] -= learning_rate * gradient


def predict_lightfm(item_features, user_features, item_biases, user_biases,
                   interactions, lightfm_data, num_threads):
    """
    Pure Python implementation of prediction.
    Simplified version of the Cython predict_lightfm function.
    """
    predictions = []
    
    for idx in range(len(interactions.data)):
        user_id = interactions.row[idx] if idx < len(interactions.row) else 0
        item_id = interactions.col[idx] if idx < len(interactions.col) else 0
        
        prediction = compute_prediction(
            item_features, item_biases,
            user_features, user_biases,
            item_id, user_id
        )
        
        predictions.append(prediction)
    
    return predictions


def predict_ranks(item_features, user_features, test_interactions, train_interactions,
                 ranks_data, lightfm_data, num_threads):
    """
    Pure Python implementation of rank prediction.
    Simplified version of the Cython predict_ranks function.
    """
    print("Warning: Using simplified pure Python predict_ranks implementation")
    
    # Fill the ranks_data array with dummy values
    for idx in range(len(ranks_data)):
        ranks_data[idx] = 1.0  # Dummy rank


def calculate_auc_from_rank(ranks, test_interactions, num_threads):
    """
    Pure Python implementation of AUC calculation.
    Simplified version of the Cython calculate_auc_from_rank function.
    """
    print("Warning: Using simplified pure Python calculate_auc_from_rank implementation")
    
    # Return dummy AUC for now
    return 0.5


def predict_sample_bias(item_biases, user_biases, interactions, num_threads):
    """
    Pure Python implementation of sample bias prediction.
    """
    predictions = []
    
    for idx in range(len(interactions.data)):
        user_id = interactions.row[idx] if idx < len(interactions.row) else 0
        item_id = interactions.col[idx] if idx < len(interactions.col) else 0
        
        prediction = 0.0
        if hasattr(item_biases, '__getitem__') and item_id < len(item_biases):
            prediction += item_biases[item_id]
        if hasattr(user_biases, '__getitem__') and user_id < len(user_biases):
            prediction += user_biases[user_id]
            
        predictions.append(prediction)
    
    return predictions
