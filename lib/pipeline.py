import os
from prepare import download_data
from processing import crop_face
import cv2


def process_data(remove_images: bool = True):
    """
    Download data and process all the image. Now just crop face.

    Parameters
    ----------
    remove_images: bool. Default True
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
    for i, image_filename in enumerate(os.listdir(images_path)):
        image_path = os.path.join(images_path, image_filename)
        image = cv2.imread(image_path)
        state, cropped_image = crop_face(image, 1.12, 9)
        if i % 20000 == 0:
            print(f"Done {i // 2000}%")
        if state == "one":
            cv2.imwrite(image_path, cropped_image)
        else:
            remove_list.append(image_path)
    if remove_images:
        print(
            f"Delete {len(remove_list)} images, dues to found none or multiple faces in each of them."
        )
        for remove_path in remove_list:
            os.remove(os.path.join(images_path, remove_path))
    print("Done prepare dataset. Now you can build embedding query.")


if __name__ == "__main__":
    process_data()
