import torch
from tqdm import tqdm
from triplet_method import triplet_loss

def fine_tune_with_identity(model, dataloader, optimizer, num_epochs, device, num_threads=4):
    """
    Fine-Tunes the provided model using identity labels through triplet loss.
    This function performs fine-tuning of a given model by iterating over the provided 
    dataloader for a specified number of epochs. It uses triplet loss to optimize the 
    model's embeddings, ensuring that anchor samples are closer to positive samples 
    than to negative samples in the embedding space.
        model (torch.nn.Module): The neural network model to be fine-tuned.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data. 
            Each batch should contain anchor, positive, and negative samples.
        optimizer (torch.optim.Optimizer): Optimizer used to update the model parameters.
        num_epochs (int): Number of epochs to perform during fine-tuning.
        device (torch.device): Device on which the training will be performed (e.g., CPU or GPU).
        num_threads (int, optional): Number of threads to use for intra-op parallelism. Defaults to 4.
    Raises:
        ValueError: If the dataloader does not provide batches with at least three elements 
            (anchor, positive, negative).
    Example:
        ```python
        # Assume TripletDataset and triplet_loss are predefined
        dataset = TripletDataset(...)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = MyModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        fine_tune_with_identity(model, dataloader, optimizer, num_epochs=10, device=device)
        ```
    Notes:
        - Ensure that the dataloader yields batches in the format (anchor, positive, negative).
        - The model should output embeddings suitable for triplet loss computation.
        - Adjust `num_threads` based on the available CPU resources for optimal performance.
    """
    torch.set_num_threads(num_threads)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            anchor, positive, negative = batch[:3]
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Forward Pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            # Compute Triplet Loss
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Triplet Loss: {running_loss / len(dataloader):.4f}")
