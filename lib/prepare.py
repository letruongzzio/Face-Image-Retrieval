import kagglehub


def download_data() -> str:
    try:
        return kagglehub.dataset_download("jessicali9530/celeba-dataset")
    except:
        return "/home/tan/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2"


if __name__ == "__main__":
    # /home/tan/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2
    print(download_data())
