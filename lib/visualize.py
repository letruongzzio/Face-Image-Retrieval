import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
from prepare import download_data


def plot_face_with_bounding_box(image_index: int):
    """
    Plot the image with its bounding box.
    The bounding box is provided in the format of (top_left_x, top_left_y, width, height).
    You can use this by passing the image index, for example, 1, then it will plot the first image in the celebA dataset, along with its bounding box

    Parameters:
    -----------
    image_index (int):
        The index of the image, start with 1.

    Returns:
        None. DO NOT call plt.show() after calling the function.
    """
    dataset_path = download_data()
    image_dir_path = os.path.join(dataset_path, "img_align_celeba", "img_align_celeba")
    image_path = os.path.join(image_dir_path, f"{image_index:06d}.jpg")
    image = Image.open(image_path)

    bounding_box_path = os.path.join(dataset_path, "list_bbox_celeba.csv")
    bounding_box = pd.read_csv(bounding_box_path)
    bounding_box_list: list[float] = bounding_box.iloc[0][
        ["x_1", "y_1", "width", "height"]
    ].tolist()

    ax = plt.subplot(111)

    ax.imshow(image)
    bounding_box = patches.Rectangle(
        (
            bounding_box_list[0] - bounding_box_list[2] / 2,
            bounding_box_list[1] - bounding_box_list[3] / 2,
        ),
        bounding_box_list[2],
        bounding_box_list[3],
        linewidth=3,
        edgecolor="r",
        facecolor="none",
    )

    ax.add_patch(bounding_box)
    plt.show()


if __name__ == "__main__":
    plot_face_with_bounding_box(3)
