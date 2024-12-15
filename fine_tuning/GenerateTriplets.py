import random
from collections import defaultdict
import os
from concurrent.futures import ThreadPoolExecutor

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
IMAGE_DIR = os.path.join(PARENT_DIRNAME, "data/img_align_celeba")
IDENTITY_FILE_PATH = os.path.join(PARENT_DIRNAME, "data/identity_CelebA.txt")

def load_identity_labels(file_path):
    """
    Load identity labels from identity_CelebA.txt.

    Args:
        - file_path: Path to identity_CelebA.txt.

    Returns:
        - identity_labels: Dictionary {'img_name': identity}.
    """
    identity_labels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            img_name, identity = line.strip().split()
            img_path = os.path.join(IMAGE_DIR, img_name)
            if not os.path.exists(img_path):  # Check if image exists
                print(f"Warning: Image {img_name} not found. Skipping.")
                continue
            identity_labels[img_name] = int(identity)
    return identity_labels

def process_identity(identity, identity_dict, train_identities):
    """
    Generate triplets for a single identity.

    Args:
        - identity: Identity to process.
        - identity_dict: Dictionary mapping identities to images.
        - train_identities: List of training identities.

    Returns:
        - List of triplets for the given identity.
    """
    triplets = []
    images = [img for img in identity_dict[identity] if os.path.exists(os.path.join(IMAGE_DIR, img))]
    if len(images) < 2:
        return triplets  # Skip if less than two images
    for i in range(len(images) - 1):
        anchor = images[i]
        positive = images[i + 1]
        negative_identity = random.choice([id_ for id_ in train_identities if id_ != identity])
        negative_images = [img for img in identity_dict[negative_identity] if os.path.exists(os.path.join(IMAGE_DIR, img))]
        if not negative_images:
            continue
        negative = random.choice(negative_images)
        triplets.append((anchor, positive, negative))
    return triplets

def generate_triplets(identity_labels, train_ratio=0.8, num_threads=4):
    """
    Generate train_triplets, test_query_triplets, and test_gallery_triplets with multithreading.

    Args:
        - identity_labels: Dictionary {'img_name': identity}.
        - train_ratio: Ratio of identities used for training.
        - num_threads: Number of threads to use.

    Returns:
        - train_triplets: List of (Anchor, Positive, Negative) for training.
        - test_query_triplets: List of (Anchor,) for querying in test.
        - test_gallery_triplets: List of (Gallery,) for gallery in test.
    """
    identity_dict = defaultdict(list)
    for image_name, identity in identity_labels.items():
        identity_dict[identity].append(image_name)

    identities = list(identity_dict.keys())
    random.shuffle(identities)

    # Train-Test Split
    train_identities = identities[:int(train_ratio * len(identities))]
    test_identities = identities[int(train_ratio * len(identities)):]

    # Generate Train Triplets using threads
    train_triplets = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda id_: process_identity(id_, identity_dict, train_identities), train_identities)
        for triplets in results:
            train_triplets.extend(triplets)

    # Generate Test Query and Gallery
    test_query_triplets = []
    test_gallery_triplets = []
    for identity in test_identities:
        images = identity_dict[identity]
        if len(images) < 2:
            continue
        query = random.choice(images)
        test_query_triplets.append((query,))
        gallery = [img for img in images if img != query]
        test_gallery_triplets.extend([(img,) for img in gallery])

    return train_triplets, test_query_triplets, test_gallery_triplets

def GenerateTriplets(num_threads=4):
    """
    Generate triplets for training and testing using multithreading.

    Args:
        - num_threads: Number of threads to use.

    Returns:
        - train_triplets: List of (Anchor, Positive, Negative) for training.
        - test_query_triplets: List of (Anchor,) for querying in test.
        - test_gallery_triplets: List of (Gallery,) for gallery in test.
    """
    identity_labels = load_identity_labels(IDENTITY_FILE_PATH)
    train_triplets, test_query_triplets, test_gallery_triplets = generate_triplets(identity_labels, num_threads=num_threads)
    print(f"Number of train triplets: {len(train_triplets)}")
    print(f"Number of test query triplets: {len(test_query_triplets)}")
    print(f"Number of test gallery triplets: {len(test_gallery_triplets)}")

    return train_triplets, test_query_triplets, test_gallery_triplets
