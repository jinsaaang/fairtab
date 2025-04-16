import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, loss_fn=None):
        """
        A generic trainer for PyTorch classification models.

        Args:
        - model: PyTorch nn.Module
        - loss_fn: optional custom loss function (takes model, x, y, group)
        """
        self.model = model
        self.loss_fn = loss_fn or self.default_loss_fn

    def default_loss_fn(self, model, x, y):
        output = model(x)
        # log_probs = torch.log_softmax(output, dim=1)
        # return nn.NLLLoss()(log_probs, y.long())
        return nn.BCEWithLogitsLoss()(output[:, 1].squeeze(), y.float()) 

    def train(self, data_loader, params, device="cuda"):
        """
        Train the model using the provided data loader and training parameters.

        Args:
        - data_loader: DataLoader yielding (x, y, group, _) tuples
        - params: dict with 'lr' and 'epochs'
        - device: torch device
        """
        self.model.to(device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])

        for epoch in range(params["epochs"]):
            total_loss = 0
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                loss = self.loss_fn(self.model, x_batch, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f"[Train] Epoch {epoch + 1}/{params['epochs']}, Loss: {avg_loss:.4f}")

    def predict(self, data_loader, device="cuda"):
        """
        Predict labels for a given data loader.

        Args:
        - data_loader: DataLoader yielding (x, y, group, _) tuples
        - device: torch device

        Returns:
        - predictions: Tensor of predicted labels
        """
        self.model.to(device)
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for x_batch, _, _, _ in data_loader:
                x_batch = x_batch.to(device)
                output = self.model(x_batch)
                preds = torch.argmax(output, dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds)
