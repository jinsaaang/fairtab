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

    # Absolute error for classification
    # print(f"y_pred: {y_pred.shape}, y_true: {y_true.shape}")
    error = torch.abs(y_pred - y_true).unsqueeze(1)  # (N, 1)
    return error

def adjust_error(train_error, valid_error, y_train, y_valid, q_train, q_valid, g_train, 
                 num_classes=2, alpha=1.0, beta=1.0, eps=1e-8):
    """
    Adjust errors using cluster-wise label imbalance and sample scarcity,
    accounting for per-sample label frequency within clusters.

    Args:
    - train_error: (N_train, 1) torch tensor
    - valid_error: (N_valid, 1) torch tensor
    - q_train: (N_train, G) numpy array
    - q_valid: (N_valid, G) numpy array
    - g_train: (N_train,) torch tensor
    - y_train: (N_train,) torch tensor
    - y_valid: (N_valid,) torch tensor
    - num_classes: number of classes
    - alpha, beta: regularization weights
    - eps: small value to prevent division by zero

    Returns:
    - train_error_adj: (N_train, 1) torch tensor
    - valid_error_adj: (N_valid, 1) torch tensor
    """
    device = train_error.device
    N_train, G = q_train.shape

    # 1. Cluster-wise label stats
    cluster_stats = {}
    for g in range(G):
        mask = (g_train == g)
        if mask.sum() == 0:
            class_ratio = torch.ones(num_classes, device=device) / num_classes
            size_penalty = torch.tensor(1.0, device=device)
        else:
            labels_in_cluster = y_train[mask]
            class_counts = torch.bincount(labels_in_cluster, minlength=num_classes).float()
            total = class_counts.sum()
            class_ratio = class_counts / (total + eps)
            size_penalty = torch.log(1 + total) / torch.log(torch.tensor(N_train + 1.0, device=device))

        cluster_stats[g] = {
            'class_ratio': class_ratio,
            'size_factor': size_penalty
        }

    # 2. Adjust error sample-wise
    def compute_adjusted_error(error, q_np, y):
        N = error.shape[0]
        adjusted = torch.zeros_like(error, dtype=torch.float)

        for g in range(G):
            weight_np = q_np[:, g].reshape(-1, 1)  # (N, 1) numpy
            weight = torch.from_numpy(weight_np).to(device=device, dtype=error.dtype)

            class_ratio = cluster_stats[g]['class_ratio']
            size_factor = cluster_stats[g]['size_factor']

            sample_class_ratio = class_ratio[y]  # (N,)
            label_penalty = 1.0 / (sample_class_ratio + eps)
            mod = 1.0 - (alpha * (1 - label_penalty.unsqueeze(1)) + beta * (1 - size_factor))
            mod = torch.clamp(mod, min=0.1, max=2.0)

            adjusted += weight * (error * mod)

        return adjusted

    train_error_adj = compute_adjusted_error(train_error, q_train, y_train)
    valid_error_adj = compute_adjusted_error(valid_error, q_valid, y_valid)

    return train_error_adj, valid_error_adj