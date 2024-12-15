import torch
from tqdm import tqdm
from TripletMethod import triplet_loss

def fine_tune_with_identity(model, dataloader, optimizer, num_epochs, device, num_threads=4):
    """
    Fine-tune the model using identity labels.
    
    Args:
        - model: Model to fine-tune.
        - dataloader: DataLoader for the dataset.
        - optimizer: Optimizer for the model.
        - num_epochs: Number of epochs to train.
        - device: Device to run the model.
    """
    torch.set_num_threads(num_threads)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            anchor, positive, negative = batch[:3]
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Forward Pass
            anchor_emb, _ = model(anchor)
            positive_emb, _ = model(positive)
            negative_emb, _ = model(negative)

            # Compute Triplet Loss
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Triplet Loss: {running_loss / len(dataloader):.4f}")


def fine_tune_with_attributes(model, dataloader, optimizer, criterion, num_epochs, device, num_threads=4):
    """
    Fine-tune the model using attributes labels.

    Args:
        - model: Model to fine-tune.
        - dataloader: DataLoader for the dataset.
        - optimizer: Optimizer for the model.
        - criterion: Loss function for the attributes.
        - num_epochs: Number of epochs to train.
        - device: Device to run the model.
    """
    torch.set_num_threads(num_threads)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, _, _, attributes, _, _ = batch
            images, attributes = images.to(device), attributes.to(device)

            # Forward Pass
            _, attribute_outputs = model(images)

            # Compute Attributes Loss
            loss = criterion(attribute_outputs, attributes)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Attribute Loss: {running_loss / len(dataloader):.4f}")
