import csv
import os

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
IMAGE_DIR = os.path.join(PARENT_DIRNAME, "data/img_align_celeba/")
ATTR_DIRNAME = os.path.join(PARENT_DIRNAME, "data/list_attr_celeba.csv")

def process_attributes_csv():
    """
    Process the attributes CSV file and convert -1/1 to 0/1.

    Returns:
        - attributes_dict: Dictionary of attributes for each image.
    """
    attributes_dict = {}
    with open(ATTR_DIRNAME, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header
        for row in reader:
            img_name = row[0]
            img_path = os.path.join(IMAGE_DIR, img_name)
            if not os.path.exists(img_path):  # Check if image exists
                print(f"Warning: Image {img_name} not found. Skipping.")
                continue
            attributes = [int(attr) for attr in row[1:]]
            attributes_binary = [(1 if attr == 1 else 0) for attr in attributes]
            attributes_dict[img_name] = attributes_binary

    return attributes_dict
