import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub
from prepare import download_data


def upload_dataset(dataset_dir):
    metadata_path = os.path.join(dataset_dir, "dataset-metadata.json")
    handle = "daoxuantan/my-celeba"
    if not os.path.exists(metadata_path):
        kagglehub.dataset_upload(handle, dataset_dir)
    else:
        kagglehub.dataset_upload(
            handle, dataset_dir, version_notes="Update dataset automatically."
        )


if __name__ == "__main__":
    dataset_dir = download_data()
    upload_dataset(dataset_dir)
