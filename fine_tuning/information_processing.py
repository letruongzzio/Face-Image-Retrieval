import pandas as pd

def load_partitions(partition_file):
    """
    Load partitions from a csv file.

    Args:
        partition_file: str, path to the csv file containing the partitions.

    Returns:
        partitions: dict, keys are file names and values are the partition types.
    """
    df = pd.read_csv(partition_file)
    partitions = dict(zip(df["image_id"], df["partition"]))
    print(f"Loaded {len(partitions)} images with partitions.")
    return partitions

def load_attributes(attr_file):
    """
    Load attributes from a csv file.

    Args:
        attr_file: str, path to the csv file containing the attributes.

    Returns:
        full_attributes: dict, keys are file names and values are the attributes.
        attr_names: list, names of the attributes.
    """
    df = pd.read_csv(attr_file)
    file_names = df.iloc[:, 0].values
    attributes = df.iloc[:, 1:].values
    attr_names = df.columns[1:]
    full_attributes = dict(zip(file_names, attributes))
    print(f"Loaded {len(full_attributes)} images with {len(attr_names)} attributes.")
    return full_attributes, attr_names
            