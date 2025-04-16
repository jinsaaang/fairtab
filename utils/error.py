import torch

def compute_error(y_pred, y_true):
    """
    Compute sample-wise prediction error (e.g., absolute error for classification).

    Args:
    - y_pred: (N,) torch.Tensor or numpy (predicted labels or probabilities)
    - y_true: (N,) torch.Tensor or numpy (ground-truth labels)

    Returns:
    - error: (N, 1) torch.Tensor
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu()

    # Convert to 0-1 error (classification)
    error = (y_pred != y_true).float().unsqueeze(1)
    return error

def adjust_error(train_error, valid_error, train_group_labels, train_group, valid_group):
    """
    Adjust error per sample based on soft cluster-level error means.

    Args:
    - train_error: (N_train, 1) torch.Tensor
    - valid_error: (N_valid, 1) torch.Tensor
    - train_group_labels: (N_train,) numpy or tensor (hard label)
    - train_group: (N_train, G) soft group probabilities
    - valid_group: (N_valid, G) soft group probabilities

    Returns:
    - train_error_adj: (N_train, 1)
    - valid_error_adj: (N_valid, 1)
    """
    device = train_error.device

    # Compute cluster-level average error from valid set (soft weighted)
    G = valid_group.shape[1]
    cluster_error_sum = torch.zeros(G, device=device)
    cluster_weight_sum = torch.zeros(G, device=device)

    for g in range(G):
        weights = valid_group[:, g]  # (N_valid,)
        cluster_error_sum[g] = torch.sum(weights * valid_error.squeeze())
        cluster_weight_sum[g] = torch.sum(weights)

    cluster_avg_error = cluster_error_sum / (cluster_weight_sum + 1e-8)  # (G,)

    # Apply to both train and valid set
    train_error_adj = cluster_avg_error[train_group_labels].unsqueeze(1)
    valid_error_adj = torch.matmul(valid_group, cluster_avg_error.unsqueeze(1))  # (N_valid, 1)

    return train_error_adj, valid_error_adj