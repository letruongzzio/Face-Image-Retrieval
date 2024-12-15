import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def plot_retrieved_images(query_image_path, gallery_image_paths, top_k=5):
    """
    Plot the query image alongside the top-K retrieved images in a clean layout.

    Args:
        - query_image_path: Path to the query image.
        - gallery_image_paths: List of gallery image paths.
        - top_k: Number of retrieved images to display.
    """
    # Validate the number of gallery images
    top_k = min(top_k, len(gallery_image_paths))

    # Create a figure for the plot with adjustable grid
    plt.figure(figsize=(12, 6))  # Adjust width and height for better appearance
    grid_cols = top_k + 1  # Query image + top_k retrieved images
    grid_rows = 1

    # Plot the Query Image
    plt.subplot(grid_rows, grid_cols, 1)
    query_img = mpimg.imread(query_image_path)
    plt.imshow(query_img)
    plt.title("Query Image", fontsize=14, fontweight="bold", color="darkblue")
    plt.axis("off")
    plt.gca().set_facecolor("#e8f4f8")  # Add a light background for better focus

    # Plot the Retrieved Images
    for i in range(top_k):
        gallery_img = mpimg.imread(gallery_image_paths[i])
        plt.subplot(grid_rows, grid_cols, i + 2)
        plt.imshow(gallery_img)
        plt.title(f"Rank {i + 1}", fontsize=12, color="green")
        plt.axis("off")
        plt.gca().set_facecolor("#f8f8f8")  # Subtle background for retrieved images

    # Add space between plots for clarity
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing between plots
    plt.tight_layout()
    plt.show()