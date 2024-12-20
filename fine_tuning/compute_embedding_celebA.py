import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
IMAGE_DIR = os.path.join(PARENT_DIRNAME, "data/img_align_celeba/")
IMAGE_SIZE = 218

def compute_embeddings_from_images(model, device, image_dir=IMAGE_DIR, transform=None, batch_size=64):
    """
    Computes embeddings for all images in a directory.

    Args:
        - model: The trained model for embedding computation.
        - image_dir: Directory containing images.
        - device: The device to perform computations on (e.g., 'cuda' or 'cpu').
        - transform: Transformations to apply to the images.
        - batch_size: Batch size for processing images.

    Returns:
        - embeddings: Numpy array containing all embeddings.
        - filenames: List of image filenames corresponding to the embeddings.
    """
    model.eval()

    # Prepare transformations
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    embeddings = []
    filenames = []

    # List all image files in the directory
    image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]

    with torch.no_grad():
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []

            for file in batch_files:
                try:
                    # Load and preprocess the image
                    img = Image.open(file).convert("RGB")
                    img = transform(img)
                    batch_images.append(img)
                    filenames.append(file)
                except (IOError, OSError):
                    continue

            if not batch_images:
                continue

            # Stack images into a batch and move to device
            batch_images = torch.stack(batch_images).to(device)

            # Compute embeddings
            emb = model(batch_images)
            embeddings.append(emb.cpu().numpy())

    # Combine all embeddings into a single numpy array
    embeddings = np.vstack(embeddings)

    return embeddings, filenames
