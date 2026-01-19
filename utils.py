import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_images_row(images, titles=None, figsize=(15, 5)):
    """
    Display cv2 images in one row with shared axes and interactive zoom.

    Parameters
    ----------
    images : list of np.ndarray
        List of images loaded with cv2.
    titles : list of str, optional
        Titles for each subplot.
    figsize : tuple, optional
        Figure size.
    """
    if type(images) is not list:
        images = [images]
    n = len(images)
    if titles is None:
        titles = [""] * n
    if len(titles) != n:
        raise ValueError("titles must match number of images")

    fig, axes = plt.subplots(
        1, n, figsize=figsize, sharex=True, sharey=True
    )

    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        cmap = None
        disp = img

        # Infer image type
        if img.ndim == 3 and img.shape[2] == 3:
            # Color image (BGR → RGB)
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            unique_vals = np.unique(img)
            if unique_vals.size <= 2:
                cmap = "binary"
            else:
                cmap = "gray"
        else:
            raise ValueError("Unsupported image format")

        ax.imshow(disp, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
