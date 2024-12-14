import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def evaluate_model_with_accuracy(model, data_loader, criterion, device):
    """
    Evaluate the model on the given data and calculate accuracy.

    Args:
        model: PyTorch model to evaluate.
        data_loader: PyTorch DataLoader object.
        criterion: Loss function to use.
        device: Device to run the model on.

    Returns:
        avg_loss: Average loss over the data.
        accuracy: Accuracy of the model on the data.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = (outputs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def train_model_attribution(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    """
    Train the model on the given data.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch DataLoader object for training data.
        test_loader: PyTorch DataLoader object for test data.
        criterion: Loss function to use.
        optimizer: Optimizer to use.
        num_epochs: Number of epochs to train for.
        device: Device to run the model on.

    Returns:
        model: Trained PyTorch model.
        history: Dictionary containing train and test loss and accuracy.
    """
    history = {"train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = batch
            if images is None or labels is None:
                continue
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Prediction
            preds = (outputs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        # Calculate loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_predictions / total_predictions

        # Evaluate on test data
        test_loss, test_accuracy = evaluate_model_with_accuracy(model, test_loader, criterion, device)

        # Save history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return model, history

def plot_training_history(history):
    """
    Plot the training and test loss and accuracy.

    Args:
        history: Dictionary containing train and test loss and accuracy.
    """
    # Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title("Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.title("Accuracy During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()