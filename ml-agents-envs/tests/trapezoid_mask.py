import numpy as np
import matplotlib.pyplot as plt


def create_trapezoidal_mask(
        side_length: int,
        top_width: int,
        down_width: int,
        trapezoid_height: int,
) -> np.ndarray:
    """
    Create a trapezoidal mask for a square observation, with the trapezoid's bottom side aligned to the bottom of the observation.

    Args:
        side_length (int): Side length of the square observation (height and width of the mask).
        top_width (int): Width of the trapezoid at the top.
        down_width (int): Width of the trapezoid at the bottom.
        trapezoid_height (int): Height of the trapezoid.

    Returns:
        np.ndarray: A binary mask (2D array) with the trapezoidal region filled with 1s.
    """
    assert top_width < side_length
    assert down_width <= side_length
    assert trapezoid_height < side_length
    assert top_width <= down_width

    # Initialize the mask with zeros
    mask = np.zeros((side_length, side_length), dtype=np.uint8)

    # Calculate the vertical positions for the trapezoid
    bottom_y = side_length  # Bottom edge of the observation
    top_y = bottom_y - trapezoid_height  # Top edge of the trapezoid

    # Calculate the horizontal positions for the top and bottom edges of the trapezoid
    top_left = (side_length - top_width) // 2
    top_right = top_left + top_width
    bottom_left = (side_length - down_width) // 2
    bottom_right = bottom_left + down_width

    # Fill in the trapezoidal area
    for y in range(top_y, bottom_y):
        # Interpolate the width of the trapezoid at the current height
        alpha = (y - top_y) / trapezoid_height
        current_left = int((1 - alpha) * top_left + alpha * bottom_left)
        current_right = int((1 - alpha) * top_right + alpha * bottom_right + 1)
        current_right = min(current_right, side_length)

        # Fill the row in the trapezoidal range
        mask[y, current_left:current_right] = 255

    return mask


if __name__ == '__main__':
    m = create_trapezoidal_mask(side_length=16, top_width=2, down_width=13, trapezoid_height=11)

    print(m)
    plt.imshow(m, cmap='gray')
    plt.tight_layout()
    plt.show()


