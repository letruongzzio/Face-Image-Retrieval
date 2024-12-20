import os
from concurrent.futures import ThreadPoolExecutor
from prepare import download_data
from processing import crop_face
import cv2


def process_data(remove_images: bool = False):
    """
    Download data and process all the image. Now just crop face.

    Parameters
    ----------
    remove_images: bool. Default False
        A flag to remove all images that detected no face, or multiple faces

    Return
    ------
    None
    """
    print("Downloading Data...")
    dataset_path = download_data()
    print("Downloaded Data")
    images_path = os.path.join(dataset_path, "img_align_celeba", "img_align_celeba")
    remove_list = []
    print("Begin cropping face")

    def process_image(image_filename):
        image_path = os.path.join(images_path, image_filename)
        image = cv2.imread(image_path)
        state, cropped_image = crop_face(image, 1.12, 9)
        if state == "one":
            cv2.imwrite(image_path, cropped_image)
            return None
        else:
            return image_path

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        remove_list = list(executor.map(process_image, os.listdir(images_path)))

    remove_list = [path for path in remove_list if path is not None]

    if remove_images:
        print(
            f"Delete {len(remove_list)} images, dues to found none or multiple faces in each of them."
        )
        for remove_path in remove_list:
            os.remove(os.path.join(images_path, remove_path))
    else:
        print(
            f"There are {len(remove_list)} images that can be cropped effectively. They won't be dropped."
        )
    print("Done prepare dataset. Now you can build embedding query.")


if __name__ == "__main__":
    process_data()
