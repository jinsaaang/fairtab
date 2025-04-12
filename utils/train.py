import torch
import torch.nn as nn
import torch.optim as optim

def evaluate(model, dataloader, dataset_name="Test", device="cuda"):
    
    outputs, labels = train_or_eval_model(model, dataloader, {}, device, mode="eval")

    criterion = nn.NLLLoss(reduction="none")
    losses = criterion(outputs, labels).squeeze() 

    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    total_loss = losses.sum().item() / total

    print(f"\n{dataset_name} Evaluation - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    group_correct = {}
    group_total = {}
    group_loss_sum = {}

    for i in range(total):
        group = dataloader.dataset.groups[i].item()  

        if group not in group_correct:
            group_correct[group] = 0
            group_total[group] = 0
            group_loss_sum[group] = 0.0
        
        group_total[group] += 1
        group_loss_sum[group] += losses[i].item()

        if predictions[i] == labels[i]:
            group_correct[group] += 1

    group_losses = {}
    for group in sorted(group_total.keys()):
        if group_total[group] > 0:
            group_acc = group_correct[group] / group_total[group]
            group_loss_avg = group_loss_sum[group] / group_total[group]
            group_losses[group] = group_loss_avg

            print(f"  Group {group} Accuracy: {group_acc:.4f} ({group_correct[group]}/{group_total[group]})")
        else:
            print(f"  Group {group} Accuracy: N/A (No samples)")
            group_losses[group] = None  

    return accuracy, group_losses  

def train_or_eval_model(model, data_loader, params, device, mode="train", loss_fn=None):
    model.to(device)

    if mode == "train":
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.NLLLoss()
        epochs = params["epochs"]
        
        if loss_fn is None:
            criterion = nn.NLLLoss()
            
            def loss_fn(model, x, y, group=None):
                output = model(x)
                log_probs = torch.log_softmax(output, dim=1)
                return criterion(log_probs, y)
        
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch, group_batch, _ in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                group_batch = group_batch.to(device)

                optimizer.zero_grad()
                output = model(x_batch)
                loss = loss_fn(model, x_batch, y_batch, group_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(data_loader):.4f}")

        return model  

    elif mode == "eval":
        model.eval()
        outputs = []
        labels = []
        with torch.no_grad():
            for x_batch, y_batch, _, _ in data_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch)
                outputs.append(output.cpu())
                labels.append(y_batch.cpu())

        return torch.cat(outputs), torch.cat(labels)  
