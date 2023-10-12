import torch
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import os

def get_supports(args:argparse.Namespace, input: torch.Tensor):
    if args.graph_type == 'distance':
        path = args.marker_dir + '/electrode_graph/adj_mx_3d.pkl'
        if not os.path.exists(path):
            raise ValueError('adjacency matrix not found')
        with open(args.adj_mat_path, 'rb') as f:
            adj_mat = pickle.load(f)
            adj_mat = adj_mat[-1]

    elif args.graph_type == 'correlation':
        with torch.no_grad():
            seq_num, num_node, seq_len = input.shape
            inputs = input.permute(1, 0, 2).reshape(num_node, seq_num*seq_len)
            inputs = np.array(inputs.cpu())
            adj_mat = np.zeros((num_node, num_node))
            for j in range(num_node):
                for k in range(num_node):
                    adj_mat[j][k] = np.correlate(inputs[j], inputs[k], mode='valid')

            if args.normalize:
                corr_x = np.sum(inputs**2, axis=-1).reshape(1, num_node)
                corr_y = np.sum(inputs**2, axis=-1).reshape(num_node, 1)
                scale = np.sqrt(corr_x * corr_y)
                adj_mat /= scale
            adj_mat = abs(adj_mat)

            if args.top_k is not None:
                k = args.top_k
                adj_mat_no_self_edge = np.copy(adj_mat)
                np.fill_diagonal(adj_mat_no_self_edge, 0)

                # Find the indices of the top-k elements in each row
                top_k_idx = np.argsort(-adj_mat_no_self_edge, axis=1)[:, :k]

                # Create a mask with the same shape as the adjacency matrix
                mask = np.zeros_like(adj_mat, dtype=bool)

                # Set the mask values based on the top-k indices
                rows, cols = np.indices(top_k_idx.shape)
                mask[rows, top_k_idx] = True

                # Symmetric mask if not directed
                if not args.directed:
                    mask = np.logical_or(mask, mask.T)
                np.fill_diagonal(mask, 1)

                # Apply the mask to the adjacency matrix
                adj_mat = adj_mat * mask    
    else:
        raise ValueError('Unknown graph type')
    
    # Convert adjacency matrix to support matrix
    supports = []
    supports_mat = []
    if args.filter_type == "laplacian":  # ChebNet graph conv
        supports_mat.append(calculate_scaled_laplacian(adj_mat, lambda_max=None))
    elif args.filter_type == "random_walk":  # Forward random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).T)
    elif args.filter_type == "dual_random_walk":  # Bidirectional random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).T)
        supports_mat.append(calculate_random_walk_matrix(adj_mat.T).T)
    else:
        supports_mat.append(calculate_scaled_laplacian(adj_mat))
    for support in supports_mat:
        supports.append(torch.FloatTensor(support.toarray()))
    
    return adj_mat, supports

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) \
        - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    """
    State transition matrix D_o^-1W in paper.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()


        