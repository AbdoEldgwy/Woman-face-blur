import cv2

def blur_box(image, bbox, ksize=(51, 51)):
    """
    Blur a rectangular region in the image.

    Args:
        image (np.ndarray): Input image (BGR).
        bbox (tuple): (x1, y1, x2, y2) bounding box.
        ksize (tuple): Kernel size for blurring.

    Returns:
        np.ndarray: Image with blurred bbox region.
    """
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, ksize, 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image


# def blur_mask(image, mask, ksize=(51, 51)):
#     """
#     Blur the regions of the image specified by the mask.

#     Args:
#         image (np.ndarray): Input image (BGR).
#         mask (np.ndarray): Binary mask indicating regions to blur.
#         ksize (tuple): Kernel size for blurring.

#     Returns:
#         np.ndarray: Image with blurred regions.
#     """
#     blurred = cv2.GaussianBlur(image, ksize, 0)
#     image[mask == 255] = blurred[mask == 255]
#     return image