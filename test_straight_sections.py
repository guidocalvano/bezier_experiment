
import numpy as np
from main import render_bezier, bezier_pixel_intercepts


def test_straight_section():

    # bezier_points = np.empty([11, 2])
    # bezier_points[0::2, 0] = np.arange(6) * 2
    # bezier_points[0::2, 1] = 0
    # bezier_points[1::2, 0] = 1 + np.arange(5) * 2
    # bezier_points[1::4, 1] = 1
    # bezier_points[3::4, 1] = -1

    bezier_points = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [1, 1],
        [0, 0]
    ]) * 20

    # bezier_pixel_intercepts(bezier_points)

    render_bezier(bezier_points, [50, 50])
    pass


if __name__ == '__main__':
    test_straight_section()
