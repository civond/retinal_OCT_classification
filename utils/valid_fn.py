import torch
from tqdm import tqdm

def valid_fn(device, loader, model, optimizer, loss_fn, scaler):
    model.eval()  # make sure model is in training mode
    loop = tqdm(loader)
    
    # Track loss and dice score
    total_loss = 0
    total_samples = 0
    total_correct = 0

    # Main loop
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)


        # forward pass
        with torch.amp.autocast('cuda'):
            #print(type(model(data))) <--- sanity check
            predictions = model(data)  # <-- extract tensor
            loss = loss_fn(predictions, labels)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        batch_size = data.size(0)
        total_loss += loss.item() * batch_size  # sum the batch contributions
        total_samples += batch_size

        preds = predictions.argmax(dim=1)       # predicted class indices
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples


    return avg_loss, avg_acc