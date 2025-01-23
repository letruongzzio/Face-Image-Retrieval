import random
from collections import defaultdict
import os
from concurrent.futures import ThreadPoolExecutor

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
IMAGE_DIR = os.path.join(PARENT_DIRNAME, "data/img_align_celeba")
IDENTITY_FILE_PATH = os.path.join(PARENT_DIRNAME, "data/identity_CelebA.txt")

def load_identity_labels(file_path):
    """
    Load identity labels from a specified file, mapping image filenames to identity numbers.

    This function reads identity labels from a designated text file. Each line in the file should contain
    an image filename and its corresponding identity number, separated by whitespace. The function verifies
    the existence of each image file within the `IMAGE_DIR`. Only existing images are included in the resulting
    dictionary. If an image file is missing, the function skips it and logs a warning message.

    Args:
        file_path (str): Path to the identity labels file. Each line should follow the format:
                         `<image_filename> <identity_number>`

    Returns:
        dict: A dictionary mapping existing image filenames (`str`) to their identity numbers (`int`).

    Example:
        Given a file `identity_CelebA.txt` with the following content:

            img_1.jpg 1
            img_2.jpg 1
            img_3.jpg 2

        If `img_2.jpg` is missing from the `IMAGE_DIR`, the function will output a warning:

            Warning: Image img_2.jpg not found. Skipping.

        The returned `identity_labels` will be:
            {
                "img_1.jpg": 1,
                "img_3.jpg": 2
            }
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
    Generate triplets for a single identity to be used in training a model.

    This function creates triplet sets consisting of an anchor, a positive example (same identity as anchor),
    and a negative example (different identity). It ensures that all images referenced in the triplets exist
    in the specified IMAGE_DIR. The function skips any missing images and only processes identities that
    have at least two valid images to form a meaningful triplet.

        identity (str): The identity for which triplets are being generated.
        identity_dict (dict): A dictionary mapping each identity to a list of associated image filenames.
        train_identities (list): A list of all identities available for training, used to select negative examples.

        list: A list of tuples, each containing three image filenames (anchor, positive, negative).

    Example:
        triplets = process_identity("person_1", identity_dict, train_identities)
        # triplets might look like:
        # [("person_1_img1.jpg", "person_1_img2.jpg", "person_2_img3.jpg"),
        #  ("person_1_img2.jpg", "person_1_img3.jpg", "person_3_img1.jpg"), ...]

    Notes:
        - Ensure that the IMAGE_DIR variable is correctly set to the directory containing all images.
        - The function relies on the availability of at least two images per identity to form valid triplets.
        - If a negative identity has no available images in IMAGE_DIR, the triplet is skipped.
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
    Generate training and testing triplets for image processing using multithreading.

    This function constructs triplets comprising an anchor, a positive example (same identity as the anchor),
    and a negative example (different identity) for training purposes. Additionally, it generates query and
    gallery pairs for testing. The use of multithreading enhances the performance by parallelizing the
    triplet generation process.

    Parameters:
        identity_labels (dict): A dictionary mapping image filenames to their corresponding identity labels.
                                Example: {'image1.jpg': 'person1', 'image2.jpg': 'person2', ...}
        train_ratio (float, optional): The proportion of identities to be used for training. Defaults to 0.8.
        num_threads (int, optional): The number of threads to utilize for parallel processing. Defaults to 4.

        tuple:
            - train_triplets (list of tuples): Each tuple contains three image filenames
              (anchor, positive, negative) for training.
            - test_galleries (list of tuples): Each tuple contains one image filename to be used as a query in testing.
            - test_galleries (list of tuples): Each tuple contains one image filename to be used in the gallery for testing.

    Example:
        >>> identity_labels = {
        ...     'img1.jpg': 'person1',
        ...     'img2.jpg': 'person1',
        ...     'img3.jpg': 'person2',
        ...     'img4.jpg': 'person3',
        ...     'img5.jpg': 'person2',
        ... }
        >>> train, test_queries, test_galleries = generate_triplets(identity_labels, train_ratio=0.6, num_threads=2)
        >>> print(train)
        [('img1.jpg', 'img2.jpg', 'img3.jpg'), ...]
        >>> print(test_queries)
        [('img4.jpg',), ...]
        >>> print(test_galleries)
        [('img5.jpg',), ...]

    Notes:
        - The function shuffles the identities before splitting them into training and testing sets.
        - Each identity must have at least two images to form a valid triplet.
        - The number of training triplets generated may be less than expected due to skipped identities.
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
    test_queries = []
    test_galleries = []
    for identity in test_identities:
        images = identity_dict[identity]
        if len(images) < 2:
            continue
        query = random.choice(images)
        test_queries.append((query,))
        gallery = [img for img in images if img != query]
        test_galleries.extend([(img,) for img in gallery])

    return train_triplets, test_queries, test_galleries

def GenerateTriplets(num_threads=4):
    """
    This function loads identity labels from a specified file and generates
    triplet data for both training and testing purposes. By utilizing multithreading,
    it efficiently processes large datasets to create anchor, positive, and negative
    samples necessary for training machine learning models, particularly in tasks
    like face recognition or metric learning.

        num_threads (int, optional): The number of threads to use for generating triplets.
            Defaults to 4.

        tuple:
            - train_triplets (List[Tuple[Any, Any, Any]]): A list of tuples in the form
                (Anchor, Positive, Negative) used for training the model.
            - test_queries (List[Tuple[Any]]): A list of tuples containing
                (Anchor,) used as query samples in the test set.
            - test_galleries (List[Tuple[Any]]): A list of tuples containing
                (Gallery,) used as gallery samples in the test set.

    Example:
        >>> train, test_queries, test_galleries = GenerateTriplets(num_threads=8)
        >>> print(f"Training triplets: {len(train)}")
        Training triplets: 50000
        >>> print(f"Test queries: {len(test_queries)}")
        Test queríes: 10000
        >>> print(f"Test galleries: {len(test_galleries)}")
        Test galleries: 5000

    Notes:
        - The function reads identity labels from a specified file and processes images
            to generate triplets for training and testing.
        - The number of training triplets may be less than expected due to skipped identities.
        - The function uses multithreading to improve the efficiency of triplet generation.
    """
    identity_labels = load_identity_labels(IDENTITY_FILE_PATH)
    train_triplets, test_queries, test_galleries = generate_triplets(identity_labels, num_threads=num_threads)
    print(f"Number of train triplets: {len(train_triplets)}")
    print(f"Number of test query triplets: {len(test_queries)}")
    print(f"Number of test gallery triplets: {len(test_galleries)}")

    return train_triplets, test_queries, test_galleries
