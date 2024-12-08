import kagglehub


def download_data() -> str:
    """
    Download the CelebA dataset using Kaggle API. Return the path of the dataset.
    """
    try:
        return kagglehub.dataset_download("jessicali9530/celeba-dataset")
    except:
        return "~/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2"


if __name__ == "__main__":
    # ~/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2
    print(download_data())
