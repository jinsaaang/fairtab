import torch
import numpy as np

def compute_error(y_pred, y_true):
    """
    Compute sample-wise prediction error (e.g., absolute error for classification).

    Args:
    - y_pred: (N,) torch.Tensor or numpy (predicted labels or probabilities)
    - y_true: (N,) torch.Tensor or numpy (ground-truth labels)

    Returns:
    - error: (N, 1) torch.Tensor
    """
    y_pred = y_pred.to(dtype=torch.float32)
    y_true = y_true.to(dtype=torch.long)

    # Absolute error for classification
    correct_probs = y_pred[torch.arange(len(y_true)), y_true]  # (N,)
    error = 1.0 - correct_probs  # (N,)
    return error.unsqueeze(1)


def adjust_error(train_error, valid_error, q_train, q_valid,
                 g_train, y_train, y_valid,
                 num_classes=2, alpha=1.0, beta=1.0, eps=1e-8):
    """
    Adjust sample-wise errors using:
    - whether the sample's class is the major class in its cluster
    - how large the cluster is (large group → error ↑)

    Args:
        train_error, valid_error: (N, 1) torch tensors
        q_train, q_valid: (N, G) torch tensors (soft assignments)
        g_train: (N,) torch tensor of hard cluster assignments
        y_train, y_valid: (N,) torch tensor of true labels
        alpha, beta: scaling factors
        num_classes: number of class labels
        eps: for numerical stability

    Returns:
        train_error_adj, valid_error_adj: torch tensors (N, 1)
    """
    train_error = train_error.to(torch.float32)
    valid_error = valid_error.to(torch.float32)

    device = train_error.device
    N_train, G = q_train.shape

    # 1. group statistics
    cluster_stats = {}
    for g in range(G):
        mask = (g_train == g)
        if mask.sum() == 0:
            class_counts = torch.ones(num_classes, device=device)
            size_score = torch.tensor(0.0, device=device)
        else:
            labels = y_train[mask]
            class_counts = torch.bincount(labels, minlength=num_classes).float().to(device)
            size_score = torch.log(torch.tensor(1.0 + labels.size(0), device=device)) / \
                         torch.log(torch.tensor(1.0 + N_train, device=device))

        major_class = torch.argmax(class_counts)
        cluster_stats[g] = {
            'major_class': major_class.item(),  # int
            'size_score': size_score.item()     # float
        }

    # 2. sample-wise adjustment
    def compute_adjusted_error(error, q_soft, y):
        N = error.shape[0]
        adj_error = torch.zeros(error.shape, device=error.device, dtype=error.dtype)

        for g in range(G):
            weight = q_soft[:, g].unsqueeze(1).to(device=device, dtype=error.dtype)  # (N, 1)

            major_class = cluster_stats[g]['major_class']
            size_score = cluster_stats[g]['size_score']

            is_major = (y == major_class).float().unsqueeze(1)  # (N, 1)
            is_minor = 1.0 - is_major

            # 조정 계수
            label_mod = (1.0 + alpha) * is_major + (1.0 - alpha) * is_minor
            size_mod = 1.0 + beta * (size_score - 0.5)

            mod = label_mod * size_mod  
            adj_error += weight * (error * mod)

        return adj_error

    # ensure q input is tensor
    if isinstance(q_train, np.ndarray):
        q_train = torch.from_numpy(q_train).to(device=train_error.device, dtype=train_error.dtype)
    if isinstance(q_valid, np.ndarray):
        q_valid = torch.from_numpy(q_valid).to(device=valid_error.device, dtype=valid_error.dtype)

    train_error_adj = compute_adjusted_error(train_error, q_train, y_train)
    valid_error_adj = compute_adjusted_error(valid_error, q_valid, y_valid)

    return train_error_adj, valid_error_adj