import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class CelebAAttributionDataset(Dataset):
    """
    Dataset class for CelebA dataset
    """
    def __init__(self, image_dir, attributes, partitions, partition_type, transform=None):
        """
        Args:
            - image_dir (string): Directory with all the images.
            - attributes (dict): Dictionary with image file name as key and attribute list as value.
            - partitions (dict): Dictionary with image file name as key and partition type as value.
            - partition_type (string): Type of partition to use. Can be 'train', 'val', or 'test'.
            - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_files = [f for f, p in partitions.items() if p == partition_type]
        self.labels = [attributes[f] for f in self.image_files]
        self.transform = transform

        # Eliminate images with None attributes
        self.eliminate_none_attributes()

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Args:
            - idx (int): Index of the image to return.
        
        Returns:
            - image (Tensor): Image of the specified index.
            - label (Tensor): Label of the specified index.
        """
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            label = (label + 1) / 2
            if self.transform:
                image = self.transform(image)
            return image, label
        except (FileNotFoundError, OSError, IOError):
            return None
    
    def eliminate_none_attributes(self):
        """
        Eliminate images with None attributes or invalid images.
        """
        valid_image_files = []
        valid_labels = []

        for idx, label in enumerate(self.labels):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            try:
                Image.open(img_path).convert("RGB")
                label_tensor = torch.tensor(label, dtype=torch.float32)
                
                # Eliminate images with None attributes
                if label is not None and not torch.isnan(label_tensor).any():
                    valid_image_files.append(self.image_files[idx])
                    valid_labels.append(label)
            except (FileNotFoundError, OSError, IOError):
                continue

        self.image_files = valid_image_files
        self.labels = valid_labels


