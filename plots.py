import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_image_with_bboxes(image: np.ndarray, bboxes: list[dict]):
    """
    Plots an image with bounding boxes and their labels.

    Args:
        image (np.ndarray): The image to be plotted.
        bboxes (list[dict]): List of dictionaries containing bbox, labels, and cropped_img.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)

    for bbox_info in bboxes:
        bbox = bbox_info["bbox"]
        labels = bbox_info["labels"]
        
        # Bounding box coordinates
        x_min, y_min,x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # Draw rectangle
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label_text = ", ".join(map(str, labels))  # Assuming labels is a list of integers/strings
        ax.text(x_min, y_min - 5, label_text, color='red', fontsize=10, backgroundcolor="white")

    plt.axis("off")
    plt.show()

    for bbox_info in bboxes:
        cropped_img = bbox_info["cropped_img"]
        fig, ax = plt.subplots(figsize=(10, 10))
        labels = bbox_info["labels"]
        ax.imshow(cropped_img, cmap='gray' if len(cropped_img.shape) == 2 else None)
        label_text = ", ".join(map(str, labels))  # Assuming labels is a list of integers/strings
        ax.text(x_min, y_min - 5, label_text, color='red', fontsize=10, backgroundcolor="white")
        plt.axis("off")
        plt.show()

