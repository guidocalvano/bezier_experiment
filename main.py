import numpy as np
import matplotlib.pyplot as plt



def bezier(bezier_points):

    DETAIL = 5
    t_values_last = (np.arange(DETAIL) / (DETAIL - 1))[np.newaxis, :, np.newaxis]
    t_values_first = 1 - t_values_last

    start = bezier_points[0:-1:2, np.newaxis, :]
    middle = bezier_points[1::2, np.newaxis, :]
    end = bezier_points[2::2, np.newaxis, :]

    a = (t_values_first * start + t_values_last * middle)
    b = (t_values_first * middle + t_values_last * end)

    p_final = np.reshape(t_values_first * a + t_values_last * b, [-1, 2])


    plt.plot(p_final[:, 0], p_final[:, 1])
    plt.show()


def bezier_as_quadratic(bezier_points):
    DETAIL = 5
    t_values_last = (np.arange(DETAIL) / (DETAIL - 1))[np.newaxis, :, np.newaxis]
    t_values_first = 1 - t_values_last

    start = bezier_points[0:-1:2, np.newaxis, :]
    middle = bezier_points[1::2, np.newaxis, :]
    end = bezier_points[2::2, np.newaxis, :]

    p_final = np.reshape(middle + t_values_first * t_values_first * (start - middle) + t_values_last * t_values_last * (end - middle), [-1, 2])

    plt.plot(p_final[:, 0], p_final[:, 1])
    plt.show()

def bezier_as_quadratic_with_one_t(bezier_points):
    start = bezier_points[0:-1:2, np.newaxis, :]
    middle = bezier_points[1::2, np.newaxis, :]
    end = bezier_points[2::2, np.newaxis, :]

    DETAIL = 50
    t_values_last = (np.arange(DETAIL) / (DETAIL - 1))[np.newaxis, :, np.newaxis]
    t_values_first = (1 - t_values_last)
    # p_final = np.reshape(middle + (1 - t_values_last) * (1 - t_values_last) * (start - middle) + t_values_last * t_values_last * (end - middle), [-1, 2])
    # p_final = np.reshape(middle + (t_values_last * t_values_last - 2 * t_values_last + 1) * (start - middle) + t_values_last * t_values_last * (end - middle), [-1, 2])
    p_final = np.reshape((t_values_last * t_values_last) * (start + end - 2 * middle) - t_values_last * 2 * (start - middle) + 1 * start, [-1, 2])

    plt.plot(p_final[:, 0], p_final[:, 1])
    plt.show()


def bezier_pixel_intercepts(bezier_points):

    p_minimum, p_maximum = bezier_maxima_minima(bezier_points)

    intercept_ranges = np.ceil(p_maximum - p_minimum)
    cummulative_intercept_count = np.cumsum(intercept_ranges, axis=0)
    intercept_offsets = cummulative_intercept_count - intercept_ranges

    intercept_count = cummulative_intercept_count[-1, :]

    intercept_indices = intercept_offsets + np.arange(intercept_count) % intercept_ranges
    intercept_values = p_minimum + np.arange(intercept_count) % intercept_ranges

    start = bezier_points[0:-1:2, np.newaxis, :]
    middle = bezier_points[1::2, np.newaxis, :]
    end = bezier_points[2::2, np.newaxis, :]

    DETAIL = 50
    t_values_last = (np.arange(DETAIL) / (DETAIL - 1))[np.newaxis, :, np.newaxis]
    t_values_first = (1 - t_values_last)
    # p_final = np.reshape(middle + (1 - t_values_last) * (1 - t_values_last) * (start - middle) + t_values_last * t_values_last * (end - middle), [-1, 2])
    # p_final = np.reshape(middle + (t_values_last * t_values_last - 2 * t_values_last + 1) * (start - middle) + t_values_last * t_values_last * (end - middle), [-1, 2])

    a = (start + end - 2 * middle)
    b = -2 * (start - middle)
    c = start

    # derivative for maximum/minimum = 0:
    # 2 * a * t + b = 0
    # t_max_or_min = -b / (2 * a)

    b_per_a = b / a
    c_per_a = c / a


    m = b_per_a / 2
    u_squared_with_impossibles = m * m - c_per_a
    u_squared = u_squared_with_impossibles[u_squared_with_impossibles > 0]
    u = np.sqrt(u_squared)

    t1 = m - u
    t2 = m + u


def bezier_maxima_minima(bezier_points):
    start = bezier_points[0:-1:2, np.newaxis, :]
    middle = bezier_points[1::2, np.newaxis, :]
    end = bezier_points[2::2, np.newaxis, :]

    DETAIL = 50
    t_values_last = (np.arange(DETAIL) / (DETAIL - 1))[np.newaxis, :, np.newaxis]
    t_values_first = (1 - t_values_last)
    # p_final = np.reshape(middle + (1 - t_values_last) * (1 - t_values_last) * (start - middle) + t_values_last * t_values_last * (end - middle), [-1, 2])
    # p_final = np.reshape(middle + (t_values_last * t_values_last - 2 * t_values_last + 1) * (start - middle) + t_values_last * t_values_last * (end - middle), [-1, 2])

    a = (start + end - 2 * middle)
    b = -2 * (start - middle)
    c = start

    # derivative for maximum/minimum = 0:
    # 2 * a * t + b = 0
    # t_max_or_min = -b / (2 * a)

    b_per_a = b / a
    c_per_a = c / a

    t_optimum = np.clip(-b_per_a / 2, 0, 1) # [:, 0, :, np.newaxis]
    p_optimum = (t_optimum * t_optimum) * (start + end - 2 * middle) - t_optimum * 2 * (start - middle) + 1 * start
    #
    minimum_p = np.minimum(np.minimum(start, end), p_optimum)[:, 0, :]
    maximum_p = np.maximum(np.maximum(start, end), p_optimum)[:, 0, :]

    p_contour = np.reshape((t_values_last * t_values_last) * (start + end - 2 * middle) - t_values_last * 2 * (start - middle) + 1 * start, [-1, 2])

    plt.plot(p_contour[:, 0], p_contour[:, 1])

    plt.scatter(minimum_p[:, 0], np.zeros_like(minimum_p[:, 0]), c="green")
    plt.scatter(maximum_p[:, 0], np.zeros_like(minimum_p[:, 0]), c="red")

    plt.scatter(np.zeros_like(minimum_p[:, 1]), minimum_p[:, 1], c="green")
    plt.scatter(np.zeros_like(maximum_p[:, 1]), maximum_p[:, 1], c="red")

    plt.show()
    return minimum_p, maximum_p


def main():

    # bezier_points = np.empty([11, 2])
    # bezier_points[0::2, 0] = np.arange(6) * 2
    # bezier_points[0::2, 1] = 0
    # bezier_points[1::2, 0] = 1 + np.arange(5) * 2
    # bezier_points[1::4, 1] = 1
    # bezier_points[3::4, 1] = -1

    cycle_points = np.empty([13, 2])
    cycle_points[:, 0] = np.arange(13)
    cycle_points[0::2, 1] = 0
    # cycle_points[1::2, 0] = 1 + np.arange(6) * 2
    cycle_points[1::4, 1] = 5
    cycle_points[3::4, 1] = -3.5

    radius = 2 + cycle_points[:, 1]
    alpha = (np.arange(13) / 12) * np.pi * 2

    bezier_points = np.empty([13, 2])
    bezier_points[:, 0] = np.cos(alpha) * radius * 2
    bezier_points[:, 1] = np.sin(alpha) * radius * 2

    bezier_pixel_intercepts(bezier_points)

    pass


if __name__ == '__main__':
    main()
