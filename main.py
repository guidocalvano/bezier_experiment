import sys

import numpy as np
import matplotlib.pyplot as plt

from roots_of_quadratic import roots_of_quadratic



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


def render_bezier(bezier_points, shape=None):

    start, middle, end = to_start_middle_end(bezier_points)
    a, b, c = to_a_b_c_quadratic(start, middle, end)
    minimum_p, maximum_p = find_optima(start, middle, end, a, b,c)

    intercept_values, intercept_bezier_indices = to_grid_intercept_xy_values_with_bezier_indices(minimum_p, maximum_p)

    t1, t2, has_intercept = to_grid_intercept_t_values(a, b, c, intercept_values, intercept_bezier_indices)

    all_t_unsorted, all_intercept_indices_by_t_unsorted = valid_intercept_t_values(t1, t2, has_intercept)

    all_t_as_cycle, all_intercept_indices_by_t_as_cycle = intercepts_as_cycle(all_t_unsorted, all_intercept_indices_by_t_unsorted, intercept_bezier_indices)

    a_bezier_indexed_by_intercept, b_bezier_indexed_by_intercept, c_bezier_indexed_by_intercept = to_bezier_indexed_by_intercept(a, b, c, intercept_bezier_indices)


    pixel_intercept, next_pixel_intercept, prior_pixel_coordinate, pixel_coordinate, next_pixel_coordinate = to_intercept_and_pixel_coordinates(a_bezier_indexed_by_intercept,
                                       b_bezier_indexed_by_intercept,
                                       c_bezier_indexed_by_intercept,
                                       all_intercept_indices_by_t_as_cycle,
                                       all_t_as_cycle
                                       )



    enter_or_leave = to_row_enter_and_leaves(prior_pixel_coordinate, pixel_coordinate, next_pixel_coordinate)
    plt.scatter(pixel_coordinate[:, 0], pixel_coordinate[:, 1], c=enter_or_leave)
    plt.show()

    filled_bitmap, outline_coordinate = bitmap_fill(pixel_coordinate, enter_or_leave, shape)

    anti_aliasing_coordinates, next_anti_aliasing_coordinates, anti_aliasing_normal = bitmap_outline_antialiasing_coordinates(
        pixel_intercept, next_pixel_intercept)

    random_sample_anti_aliasing(filled_bitmap, outline_coordinate, anti_aliasing_coordinates, anti_aliasing_normal)

    bounding_area_anti_aliasing(filled_bitmap, outline_coordinate, anti_aliasing_coordinates, next_anti_aliasing_coordinates, anti_aliasing_normal)

    return filled_bitmap

def render_bezier_contours(bezier_points, contour_count, contour_starts):

    pass




def to_start_middle_end(bezier_points):

    start = bezier_points[0:-1:2, :]
    middle = bezier_points[1::2, :]
    end = bezier_points[2::2, :]

    return start, middle, end

def to_a_b_c_quadratic(start, middle, end):

    a = (start + end - 2 * middle)
    b = -2 * (start - middle)
    c = start

    return a, b, c

def find_optima(start, middle, end, a, b, c):

    epsilon_for_optimum = (a == 0.0) * sys.float_info.epsilon


    t_optimum = np.clip(-b / (2 * a + epsilon_for_optimum), 0, 1)
    p_optimum = a * (t_optimum * t_optimum) + b * t_optimum + c

    minimum_p = np.nanmin([np.minimum(start, end), p_optimum], axis=0)[:, :]  #@TODO check if changing [:, 0, :] to [:, :] is correct
    maximum_p = np.nanmax([np.maximum(start, end), p_optimum], axis=0)[:, :]

    return minimum_p, maximum_p

def to_grid_intercept_xy_values_with_bezier_indices(p_minimum, p_maximum):
    bezier_indices = np.reshape(np.stack(np.indices(p_minimum.shape), axis=-1), [np.prod(p_minimum.shape), -1])

    intercept_ranges = np.ceil(p_maximum - p_minimum).flatten().astype(np.uint32)
    cummulative_intercept_count = np.cumsum(intercept_ranges, axis=0).astype(np.uint32)

    intercept_count = cummulative_intercept_count[-1]

    # intercept_indices = np.repeat(intercept_offsets, intercept_ranges) + np.arange(intercept_count) % np.repeat(intercept_ranges, intercept_ranges)
    intercept_bezier_indices = np.repeat(bezier_indices, intercept_ranges, axis=0)
    intercept_values = np.repeat(p_minimum.flatten().astype(np.int32), intercept_ranges) + np.arange(
        intercept_count) % np.repeat(intercept_ranges, intercept_ranges)

    return intercept_values, intercept_bezier_indices

def to_grid_intercept_t_values(a, b, c, intercept_values, intercept_bezier_indices):
    bezier_by_intercept = to_bezier_indexed_by_intercept(a, b, c, intercept_bezier_indices)

    bezier_component_by_intercept = to_bezier_component_indexed_by_interecept(*bezier_by_intercept, intercept_bezier_indices)

    t1, t2, has_intercept = t_values_for_quadratic(*bezier_component_by_intercept, intercept_values)

    return t1, t2, has_intercept


def to_bezier_indexed_by_intercept(a, b, c, intercept_bezier_indices):
    a_bezier_indexed_by_intercept = a[intercept_bezier_indices[:, 0], :]
    b_bezier_indexed_by_intercept = b[intercept_bezier_indices[:, 0], :]
    c_bezier_indexed_by_intercept = c[intercept_bezier_indices[:, 0], :]

    return a_bezier_indexed_by_intercept, b_bezier_indexed_by_intercept, c_bezier_indexed_by_intercept

def to_bezier_component_indexed_by_interecept(a_bezier_indexed_by_intercept, b_bezier_indexed_by_intercept, c_bezier_indexed_by_intercept, intercept_bezier_indices):
    a_bezier_component_indexed_by_intercept = a_bezier_indexed_by_intercept[np.arange(intercept_bezier_indices.shape[0]), intercept_bezier_indices[:, 1]]
    b_bezier_component_indexed_by_intercept = b_bezier_indexed_by_intercept[np.arange(intercept_bezier_indices.shape[0]), intercept_bezier_indices[:, 1]]
    c_bezier_component_indexed_by_intercept = c_bezier_indexed_by_intercept[np.arange(intercept_bezier_indices.shape[0]), intercept_bezier_indices[:, 1]]

    return a_bezier_component_indexed_by_intercept, b_bezier_component_indexed_by_intercept, c_bezier_component_indexed_by_intercept

def t_values_for_quadratic(a_bezier_component_indexed_by_intercept, b_bezier_component_indexed_by_intercept, c_bezier_component_indexed_by_intercept, intercept_values):
    c_with_intercepts = c_bezier_component_indexed_by_intercept - intercept_values
    # derivative for maximum/minimum = 0:
    # 2 * a * t + b = 0
    # t_max_or_min = -b / (2 * a)

    epsilon_bezier_component_indexed_by_intercept = (a_bezier_component_indexed_by_intercept == 0.0) * sys.float_info.epsilon
    b_per_a = (b_bezier_component_indexed_by_intercept / (a_bezier_component_indexed_by_intercept + epsilon_bezier_component_indexed_by_intercept))
    c_with_intercepts_per_a = (c_with_intercepts / (a_bezier_component_indexed_by_intercept + epsilon_bezier_component_indexed_by_intercept))
    # @TODO start of use of correct quadratic
    solutions, has_solutions = roots_of_quadratic(a_bezier_component_indexed_by_intercept, b_bezier_component_indexed_by_intercept, c_with_intercepts)
    return solutions[:, 0], solutions[:, 1], has_solutions

    t1, t2 = solutions[:, 0][has_solutions[:, 0]], solutions[:, 1][has_solutions[:, 1]]
    return t1, t2, has_solutions
    # end use of correct quadratic
    return solve_scaled_quadratic(b_per_a, c_with_intercepts_per_a)

def solve_scaled_quadratic(b_per_a, c_per_a):
    x_of_optimum = -b_per_a / 2
    distance_squared_intercept_to_optimum = x_of_optimum * x_of_optimum - c_per_a
    has_intercept = distance_squared_intercept_to_optimum > 0
    x_of_optimum_with_intercept = x_of_optimum[has_intercept]
    distance_to_optimum_with_intercept = np.sqrt(distance_squared_intercept_to_optimum[has_intercept])

    t1 = x_of_optimum_with_intercept - distance_to_optimum_with_intercept
    t2 = x_of_optimum_with_intercept + distance_to_optimum_with_intercept

    return t1, t2, has_intercept


def valid_intercept_t_values(t1, t2, has_intercept):
    t1_with_intercept, t2_with_intercept = t1[has_intercept[:, 0]], t2[has_intercept[:, 1]]

    t1_is_valid = np.logical_and(0 <= t1_with_intercept, t1_with_intercept <= 1)
    t2_is_valid = np.logical_and(0 <= t2_with_intercept, t2_with_intercept <= 1)

    # intercept_indices_with_valid_t = np.arange(a_bezier_indexed_by_intercept.shape[0])[u_squared_is_possible]
    # intercept_indices_with_valid_t1 = intercept_indices_with_valid_t[t1_is_valid]
    # intercept_indices_with_valid_t2 = intercept_indices_with_valid_t[t2_is_valid]
    intercept_indices = np.arange(has_intercept.shape[0])
    intercept_indices_with_valid_t1 = intercept_indices[has_intercept[:, 0]][t1_is_valid]
    intercept_indices_with_valid_t2 = intercept_indices[has_intercept[:, 1]][t2_is_valid]

    valid_t1 = t1_with_intercept[t1_is_valid]
    valid_t2 = t2_with_intercept[t2_is_valid]

    all_t_unsorted = np.concatenate([valid_t1, valid_t2])
    all_intercept_indices_by_t_unsorted = np.concatenate([intercept_indices_with_valid_t1, intercept_indices_with_valid_t2])


    # t1_is_valid = np.logical_and(0 <= t1, t1 <= 1)
    # t2_is_valid = np.logical_and(0 <= t2, t2 <= 1)
    #
    # valid_t1 = t1[t1_is_valid]
    # valid_t2 = t2[t2_is_valid]
    #
    # intercept_indices_with_valid_t = np.arange(has_intercept.shape[0])[has_intercept]
    # intercept_indices_with_valid_t1 = intercept_indices_with_valid_t[t1_is_valid]
    # intercept_indices_with_valid_t2 = intercept_indices_with_valid_t[t2_is_valid]
    #
    # all_t_unsorted = np.concatenate([valid_t1, valid_t2])
    #
    # all_intercept_indices_by_t_unsorted = np.concatenate([intercept_indices_with_valid_t1, intercept_indices_with_valid_t2])

    return all_t_unsorted, all_intercept_indices_by_t_unsorted


def intercepts_as_cycle(all_t_unsorted, all_intercept_indices_by_t_unsorted, intercept_bezier_indices):

    t_cycle_order = np.argsort(intercept_bezier_indices[all_intercept_indices_by_t_unsorted, 0]  + all_t_unsorted)

    all_t_as_cycle = all_t_unsorted[t_cycle_order]
    all_intercept_indices_by_t_as_cycle = all_intercept_indices_by_t_unsorted[t_cycle_order]

    return all_t_as_cycle, all_intercept_indices_by_t_as_cycle

def to_intercept_and_pixel_coordinates(a_bezier_indexed_by_intercept,
                                  b_bezier_indexed_by_intercept,
                                  c_bezier_indexed_by_intercept,
                                  all_intercept_indices_by_t_as_cycle,
                                  all_t_as_cycle
                                  ):
    pixel_intercept = a_bezier_indexed_by_intercept[all_intercept_indices_by_t_as_cycle, :] * (
                                                                                                          all_t_as_cycle * all_t_as_cycle)[
                                                                                              :, np.newaxis] + \
                      b_bezier_indexed_by_intercept[all_intercept_indices_by_t_as_cycle, :] * all_t_as_cycle[:,
                                                                                              np.newaxis] + \
                      c_bezier_indexed_by_intercept[all_intercept_indices_by_t_as_cycle, :]

    next_pixel_intercept = np.roll(pixel_intercept, -1, axis=0)

    # @TODO this should be faster and still accurate for positive screen coordinates with .astype(np.int32) instead of floor
    pixel_coordinate = np.floor((pixel_intercept + next_pixel_intercept) / 2.0).astype(np.int32)

    next_pixel_coordinate = np.roll(pixel_coordinate, 1, axis=0)
    prior_pixel_coordinate = np.roll(pixel_coordinate, -1, axis=0)

    plt.scatter(pixel_intercept[:, 0], pixel_intercept[:, 1], c=all_intercept_indices_by_t_as_cycle/len(all_intercept_indices_by_t_as_cycle))
    plt.show()

    return pixel_intercept, next_pixel_intercept, prior_pixel_coordinate, pixel_coordinate, next_pixel_coordinate


def to_row_enter_and_leaves(prior_pixel_coordinate, pixel_coordinate, next_pixel_coordinate):

    # enter_or_leave = 1 * ((next_pixel_coordinate[:, 1] - pixel_coordinate[:, 1]) > 0) - ((pixel_coordinate[:, 1] - prior_pixel_coordinate[:, 1]) < 0)
    #
    # return enter_or_leave
    # multiply by 90 degree rotation matrix
    normal = ((next_pixel_coordinate - prior_pixel_coordinate)[:, np.newaxis, :] @ np.array([[0, -1],[1, 0]])[np.newaxis, :, :])[:, 0, :] + \
               np.logical_and(np.all(next_pixel_coordinate == prior_pixel_coordinate, axis=1), ~np.all(next_pixel_coordinate == pixel_coordinate, axis=1))[:, np.newaxis] * (next_pixel_coordinate - pixel_coordinate)
    is_convex = (np.sum((pixel_coordinate - (next_pixel_coordinate + prior_pixel_coordinate) / 2) * normal, axis=1) <=0).astype(np.float32)
    # is_convex = (-np.ceil(next_pixel_coordinate[:, 1] - pixel_coordinate[:, 1])) == x_component_normal

    x_component_normal = normal[:, 0]
    enter_or_leave = is_convex * ((x_component_normal > 0.0) * 1 + (x_component_normal < -0.0) * -1)
    enter_or_leave_color = is_convex[:, np.newaxis] * ((x_component_normal[:, np.newaxis] > 0.0) * np.array([0, 1, 0])[np.newaxis, :] + (x_component_normal[:, np.newaxis] < -0.0) * np.array([1, 0, 0])[np.newaxis, :])

    return enter_or_leave

def bitmap_fill(pixel_coordinate, enter_or_leave, size):
    # offset = np.min(pixel_coordinate, axis=0, initial=2147483647)
    # size = np.max(pixel_coordinate, axis=0, initial=-(2147483647 - 1)) - offset + 2

    outline_coordinate = pixel_coordinate # - offset[np.newaxis, :]  # * np.abs(enter_or_leave)[:, np.newaxis]
    enter_or_leave_outline = enter_or_leave
    outline_bitmap = np.zeros(size)
    # outline_bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] += enter_or_leave_outline

    #@TODO: vectorize this
    # for i in range(outline_coordinate.shape[0]):
    #     outline_bitmap[outline_coordinate[i, 0], outline_coordinate[i, 1]] += enter_or_leave_outline[i]

    outline_bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] += enter_or_leave_outline[:]

    plt.imshow(outline_bitmap)
    plt.show()

    bitmap = np.cumsum(outline_bitmap, axis=0)
    plt.imshow(bitmap)
    plt.show()
    bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] = 1

    return bitmap, outline_coordinate

def bitmap_outline_antialiasing_coordinates(pixel_intercept, next_pixel_intercept):

    to_next_anti_aliasing_coordinate = np.clip(next_pixel_intercept - pixel_intercept, -1, 1)  # clip to prevent rounding error issues that push into the next anti aliasing pixel coordinate
    anti_aliasing_pixel_coordinate = np.floor(pixel_intercept + to_next_anti_aliasing_coordinate / 2)
    anti_aliasing_coordinates = np.clip(pixel_intercept - anti_aliasing_pixel_coordinate, 0, 1)
    next_anti_aliasing_coordinates = np.clip(anti_aliasing_coordinates + to_next_anti_aliasing_coordinate, 0, 1)

    # anti_aliasing_normal = (to_next_anti_aliasing_coordinate[:, np.newaxis, :] @ np.array([[0, -1],[1, 0]])[np.newaxis, :, :])[:, 0, :] * \
    #                        np.all(anti_aliasing_coordinates != next_anti_aliasing_coordinates, axis=1, keepdims=True) # hack to fix edge case that seems to mess up random sampling, but work fine with bounding area based calculations
    ## anti_aliasing_normal /= np.linalg.norm(anti_aliasing_normal, axis=1)[:, np.newaxis]
    anti_aliasing_normal = (to_next_anti_aliasing_coordinate[:, np.newaxis, :] @ np.array([[0, -1], [1, 0]])[np.newaxis,
                                                                                :, :])[:, 0, :]
    return anti_aliasing_coordinates, next_anti_aliasing_coordinates, anti_aliasing_normal


def random_sample_anti_aliasing(bitmap, outline_coordinate, anti_aliasing_coordinates, anti_aliasing_normal):

    random_sample_points = np.random.random([outline_coordinate.shape[0], 10000, 2])
    distance_to_line = random_sample_points @ anti_aliasing_normal[:, :, np.newaxis] - anti_aliasing_coordinates[:, np.newaxis, :] @ anti_aliasing_normal[:, :, np.newaxis]
    is_inside = (distance_to_line < 0)
    ink_area = np.mean(is_inside[:, :, 0], axis=1)
    # to_next_anti_aliasing_coordinate /= np.linalg.norm(to_next_anti_aliasing_coordinate, axis=1)[:, np.newaxis]
    #
    # normal_map = np.zeros(bitmap.shape + (3,))
    # normal_map[outline_coordinate[:, 0], outline_coordinate[:, 1], 0] = .5 + to_next_anti_aliasing_coordinate[:, 0] / 2
    # normal_map[outline_coordinate[:, 0], outline_coordinate[:, 1], 2] = .5 - to_next_anti_aliasing_coordinate[:, 0] / 2
    #
    # plt.imshow(np.flip(np.transpose(normal_map, [1, 0, 2])))
    # plt.show()

    bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] = ink_area
    plt.imshow(np.flip(bitmap.T), interpolation="none")
    plt.show()

def bounding_area_anti_aliasing(bitmap, outline_coordinate, anti_aliasing_coordinates, next_anti_aliasing_coordinates, anti_aliasing_normal):

    must_mirror = anti_aliasing_normal < 0

    ink_mirrored_anti_aliasing_coordinates = must_mirror * ( 1 - anti_aliasing_coordinates) + (1 - must_mirror) * anti_aliasing_coordinates
    next_ink_mirrored_anti_aliasing_coordinates = must_mirror * ( 1 - next_anti_aliasing_coordinates) + (1 - must_mirror) * next_anti_aliasing_coordinates

    ink_bounding_size = np.maximum(ink_mirrored_anti_aliasing_coordinates, next_ink_mirrored_anti_aliasing_coordinates)

    ink_bounding_area = np.prod(ink_bounding_size, axis=1)

    paper_bounding_size = np.maximum(1 - ink_mirrored_anti_aliasing_coordinates, 1 - next_ink_mirrored_anti_aliasing_coordinates)

    paper_bounding_area = np.prod(paper_bounding_size, axis=1)

    overlapping_bounding_area = ink_bounding_area + paper_bounding_area - 1

    ink_area = ink_bounding_area - .5 * overlapping_bounding_area

    bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] = ink_area

    plt.imshow(np.flip(bitmap.T), interpolation="none")
    plt.show()


def old_quadratic_approach(a_bezier_component_indexed_by_intercept, b_bezier_component_indexed_by_intercept, c_with_intercepts):
    epsilon_a_bezier_component_indexed_by_intercept = (a_bezier_component_indexed_by_intercept == 0.0) * sys.float_info.epsilon
    b_per_a = (b_bezier_component_indexed_by_intercept / (a_bezier_component_indexed_by_intercept + epsilon_a_bezier_component_indexed_by_intercept))
    c_with_intercepts_per_a = (c_with_intercepts / (a_bezier_component_indexed_by_intercept + epsilon_a_bezier_component_indexed_by_intercept))


    m_with_impossibles = -b_per_a / 2
    u_squared_with_impossibles = m_with_impossibles * m_with_impossibles - c_with_intercepts_per_a
    u_squared_is_possible = u_squared_with_impossibles > 0
    m = m_with_impossibles[u_squared_is_possible]
    u_squared = u_squared_with_impossibles[u_squared_is_possible]
    u = np.sqrt(u_squared)

    t1 = m - u
    t2 = m + u

    return t1, t2, u_squared_is_possible




def bezier_pixel_intercepts(bezier_points):

    p_minimum, p_maximum = bezier_maxima_minima(bezier_points)

    bezier_indices = np.reshape(np.stack(np.indices(p_minimum.shape), axis=-1), [np.prod(p_minimum.shape), -1])

    intercept_ranges = np.ceil(p_maximum - p_minimum).flatten().astype(np.uint32)
    cummulative_intercept_count = np.cumsum(intercept_ranges, axis=0).astype(np.uint32)
    # intercept_offsets = cummulative_intercept_count - intercept_ranges

    intercept_count = cummulative_intercept_count[-1]

    # intercept_indices = np.repeat(intercept_offsets, intercept_ranges) + np.arange(intercept_count) % np.repeat(intercept_ranges, intercept_ranges)
    intercept_bezier_indices = np.repeat(bezier_indices, intercept_ranges, axis=0)
    intercept_values = np.repeat(p_minimum.flatten().astype(np.int32), intercept_ranges) + np.arange(intercept_count) % np.repeat(intercept_ranges, intercept_ranges)

    start = bezier_points[0:-1:2, :]
    middle = bezier_points[1::2, :]
    end = bezier_points[2::2, :]

    a_indexed_by_bezier_component = (start + end - 2 * middle)
    b_indexed_by_bezier_component = -2 * (start - middle)
    c_indexed_by_bezier_component = start

    a_bezier_indexed_by_intercept = a_indexed_by_bezier_component[intercept_bezier_indices[:, 0], :]
    b_bezier_indexed_by_intercept = b_indexed_by_bezier_component[intercept_bezier_indices[:, 0], :]
    c_bezier_indexed_by_intercept = c_indexed_by_bezier_component[intercept_bezier_indices[:, 0], :]

    a_bezier_component_indexed_by_intercept = a_bezier_indexed_by_intercept[np.arange(intercept_bezier_indices.shape[0]), intercept_bezier_indices[:, 1]]
    b_bezier_component_indexed_by_intercept = b_bezier_indexed_by_intercept[np.arange(intercept_bezier_indices.shape[0]), intercept_bezier_indices[:, 1]]
    c_bezier_component_indexed_by_intercept = c_bezier_indexed_by_intercept[np.arange(intercept_bezier_indices.shape[0]), intercept_bezier_indices[:, 1]]

    c_with_intercepts = c_bezier_component_indexed_by_intercept - intercept_values
    # derivative for maximum/minimum = 0:
    # 2 * a * t + b = 0
    # t_max_or_min = -b / (2 * a)

    old_t1, old_t2, old_u_squared_is_possible = old_quadratic_approach(a_bezier_component_indexed_by_intercept, b_bezier_component_indexed_by_intercept, c_with_intercepts)

    solutions, has_solutions = roots_of_quadratic(a_bezier_component_indexed_by_intercept, b_bezier_component_indexed_by_intercept, c_with_intercepts)

    t1, t2 = solutions[:, 0][has_solutions[:, 0]], solutions[:, 1][has_solutions[:, 1]]

    u_squared_is_possible = np.all(has_solutions, axis=1)

    t1_is_valid = np.logical_and(0 <= t1, t1 <= 1)
    t2_is_valid = np.logical_and(0 <= t2, t2 <= 1)

    # intercept_indices_with_valid_t = np.arange(a_bezier_indexed_by_intercept.shape[0])[u_squared_is_possible]
    # intercept_indices_with_valid_t1 = intercept_indices_with_valid_t[t1_is_valid]
    # intercept_indices_with_valid_t2 = intercept_indices_with_valid_t[t2_is_valid]
    intercept_indices = np.arange(a_bezier_indexed_by_intercept.shape[0])
    intercept_indices_with_valid_t1 = intercept_indices[has_solutions[:, 0]][t1_is_valid]
    intercept_indices_with_valid_t2 = intercept_indices[has_solutions[:, 1]][t2_is_valid]

    valid_t1 = t1[t1_is_valid]
    valid_t2 = t2[t2_is_valid]

    all_t_unsorted = np.concatenate([valid_t1, valid_t2])
    all_intercept_indices_by_t_unsorted = np.concatenate([intercept_indices_with_valid_t1, intercept_indices_with_valid_t2])

    t_cycle_order = np.argsort(intercept_bezier_indices[all_intercept_indices_by_t_unsorted, 0] * 2 + all_t_unsorted)

    all_t_as_cycle = all_t_unsorted[t_cycle_order]
    all_intercept_indices_by_t_as_cycle = all_intercept_indices_by_t_unsorted[t_cycle_order]

    pixel_intercept   = a_bezier_indexed_by_intercept[all_intercept_indices_by_t_as_cycle, :] * (all_t_as_cycle * all_t_as_cycle)[:, np.newaxis] + \
                        b_bezier_indexed_by_intercept[all_intercept_indices_by_t_as_cycle, :] * all_t_as_cycle[:, np.newaxis] + \
                        c_bezier_indexed_by_intercept[all_intercept_indices_by_t_as_cycle, :]

    plt.scatter(pixel_intercept[:, 0], pixel_intercept[:, 1], c=all_intercept_indices_by_t_as_cycle/len(all_intercept_indices_by_t_as_cycle))
    # plt.scatter(pixel_intercept[:, 0], pixel_intercept[:, 1], c=all_t_as_cycle)


    next_pixel_intercept = np.roll(pixel_intercept, -1, axis=0)

    #@TODO this should be faster and still accurate for positive screen coordinates with .astype(np.int32)
    pixel_coordinate = np.floor((pixel_intercept + next_pixel_intercept) / 2.0).astype(np.int32)
    # pixel_coordinate = pixel_coordinate[~(np.all(pixel_coordinate == np.roll(pixel_coordinate, 1, axis=0), axis=1)), :]
    # pixel_coordinate = pixel_coordinate[~(np.all(pixel_coordinate == np.roll(pixel_coordinate, 1, axis=0), axis=1)), :]

    next_pixel_coordinate = np.roll(pixel_coordinate, 1, axis=0)
    prior_pixel_coordinate = np.roll(pixel_coordinate, -1, axis=0)

    # multiply by 90 degree rotation matrix
    normal = ((next_pixel_coordinate - prior_pixel_coordinate)[:, np.newaxis, :] @ np.array([[0, -1],[1, 0]])[np.newaxis, :, :])[:, 0, :] + \
               np.logical_and(np.all(next_pixel_coordinate == prior_pixel_coordinate, axis=1), ~np.all(next_pixel_coordinate == pixel_coordinate, axis=1))[:, np.newaxis] * (next_pixel_coordinate - pixel_coordinate)
    is_convex = (np.sum((pixel_coordinate - (next_pixel_coordinate + prior_pixel_coordinate) / 2) * normal, axis=1) <=0).astype(np.float32)
    # is_convex = (-np.ceil(next_pixel_coordinate[:, 1] - pixel_coordinate[:, 1])) == x_component_normal

    x_component_normal = normal[:, 0]
    enter_or_leave = is_convex * ((x_component_normal > 0.0) * 1 + (x_component_normal < -0.0) * -1)
    enter_or_leave_color = is_convex[:, np.newaxis] * ((x_component_normal[:, np.newaxis] > 0.0) * np.array([0, 1, 0])[np.newaxis, :] + (x_component_normal[:, np.newaxis] < -0.0) * np.array([1, 0, 0])[np.newaxis, :])
    # enter_or_leave_color = is_convex
    pixel_center = pixel_coordinate + .5
    plt.scatter(pixel_center[:, 0], pixel_center[:, 1], c=enter_or_leave_color)

    # pixel_intercept   = a_bezier_indexed_by_intercept[u_squared_is_possible, :][t2_is_valid, :] * (valid_t2 * valid_t2)[:, np.newaxis] + \
    #                     b_bezier_indexed_by_intercept[u_squared_is_possible, :][t2_is_valid, :] * valid_t2[:, np.newaxis] + \
    #                     c_bezier_indexed_by_intercept[u_squared_is_possible, :][t2_is_valid, :]
    #

    # plt.scatter(pixel_intercept[:, 0], pixel_intercept[:, 1], c="green")

    # pixel_intercept   = a_bezier_indexed_by_intercept[u_squared_is_possible, :][t1_is_valid, :] * (valid_t1 * valid_t1)[:, np.newaxis] + \
    #                     b_bezier_indexed_by_intercept[u_squared_is_possible, :][t1_is_valid, :] * valid_t1[:, np.newaxis] + \
    #                     c_bezier_indexed_by_intercept[u_squared_is_possible, :][t1_is_valid, :]
    #
    # plt.scatter(pixel_intercept[:, 0], pixel_intercept[:, 1], c="red")

    axes = plt.gca()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.grid()
    plt.show()

    offset = np.min(pixel_coordinate, axis=0, initial=sys.maxsize) - 1
    size = np.max(pixel_coordinate, axis=0, initial=-(sys.maxsize - 1)) - offset + 2

    outline_coordinate = pixel_coordinate - offset
    enter_or_leave_outline = enter_or_leave
    outline_bitmap = np.zeros(size)
    outline_bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] += enter_or_leave_outline

    bitmap = np.cumsum(outline_bitmap, axis=0)
    bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] = 1
    plt.imshow(np.flip(bitmap.T))
    plt.show()




    to_next_anti_aliasing_coordinate = np.clip(next_pixel_intercept - pixel_intercept, -1, 1)  # clip to prevent rounding error issues that push into the next anti aliasing pixel coordinate
    anti_aliasing_pixel_coordinate = np.floor(pixel_intercept + to_next_anti_aliasing_coordinate / 2)
    anti_aliasing_coordinates = np.clip(pixel_intercept - anti_aliasing_pixel_coordinate, 0, 1)
    next_anti_aliasing_coordinates = np.clip(anti_aliasing_coordinates + to_next_anti_aliasing_coordinate, 0, 1)

    anti_aliasing_normal = (to_next_anti_aliasing_coordinate[:, np.newaxis, :] @ np.array([[0, -1],[1, 0]])[np.newaxis, :, :])[:, 0, :] * \
                           np.all(anti_aliasing_coordinates != next_anti_aliasing_coordinates, axis=1, keepdims=True) # hack to fix edge case that seems to mess up random sampling, but work fine with bounding area based calculations
    # anti_aliasing_normal /= np.linalg.norm(anti_aliasing_normal, axis=1)[:, np.newaxis]

    random_sample_points = np.random.random([outline_coordinate.shape[0], 10000, 2])
    distance_to_line = random_sample_points @ anti_aliasing_normal[:, :, np.newaxis] - anti_aliasing_coordinates[:, np.newaxis, :] @ anti_aliasing_normal[:, :, np.newaxis]
    is_inside = (distance_to_line < 0)
    ink_area = np.mean(is_inside[:, :, 0], axis=1)
    # to_next_anti_aliasing_coordinate /= np.linalg.norm(to_next_anti_aliasing_coordinate, axis=1)[:, np.newaxis]
    #
    # normal_map = np.zeros(bitmap.shape + (3,))
    # normal_map[outline_coordinate[:, 0], outline_coordinate[:, 1], 0] = .5 + to_next_anti_aliasing_coordinate[:, 0] / 2
    # normal_map[outline_coordinate[:, 0], outline_coordinate[:, 1], 2] = .5 - to_next_anti_aliasing_coordinate[:, 0] / 2
    #
    # plt.imshow(np.flip(np.transpose(normal_map, [1, 0, 2])))
    # plt.show()

    bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] = ink_area
    plt.imshow(np.flip(bitmap.T))
    plt.show()


    must_mirror = anti_aliasing_normal < 0

    ink_mirrored_anti_aliasing_coordinates = must_mirror * ( 1 - anti_aliasing_coordinates) + (1 - must_mirror) * anti_aliasing_coordinates
    next_ink_mirrored_anti_aliasing_coordinates = must_mirror * ( 1 - next_anti_aliasing_coordinates) + (1 - must_mirror) * next_anti_aliasing_coordinates

    ink_bounding_size = np.maximum(ink_mirrored_anti_aliasing_coordinates, next_ink_mirrored_anti_aliasing_coordinates)

    ink_bounding_area = np.prod(ink_bounding_size, axis=1)

    paper_bounding_size = np.maximum(1 - ink_mirrored_anti_aliasing_coordinates, 1 - next_ink_mirrored_anti_aliasing_coordinates)

    paper_bounding_area = np.prod(paper_bounding_size, axis=1)

    too_much = ink_bounding_area + paper_bounding_area - 1

    ink_area = ink_bounding_area - .5 * too_much

    bitmap[outline_coordinate[:, 0], outline_coordinate[:, 1]] = ink_area

    plt.title('bezier pixel intercept bitmap')
    plt.imshow(np.flip(bitmap.T))
    plt.show()

    return bitmap

    pass
    # integrate(fx(t) fy'(t))
    # fx(t) = a[0] * t ** 2 + b[0] * t + c[0] + anti_aliasing_pixel_coordinate.x
    # fy(t) = a[1] * t ** 2 + b[1] * t + c[1] + anti_aliasing_pixel_coordinate.y
    # fy'(t) = 2 * a[1] * t + b[1]
    # fx(t) fy'(t) =
    #   2 * a[1] * a[0] * t ** 3 + 2 * a[1] * b[0] * t ** 2 + 2 * a[1] * (c[0] + anti_aliasing_pixel_coordinate.x) * t +
    #                                  b[1] * a[0] * t ** 2 + b[1] *  b[0] * t + b[1] * (c[0] + anti_aliasing_pixel_coordinate.x)
    # = 2 * a[1] * a[0] * t ** 3 + (2 * a[1] * b[0] + b[1] * a[0]) * t ** 2 + (2 * a[1] * (c[0] + anti_aliasing_pixel_coordinate.x) + b[1] *  b[0]

    return


import matplotlib.ticker as ticker

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

    epsilon_for_a = (a == 0.0) * sys.float_info.epsilon

    b_per_a = b / (a + epsilon_for_a)
    c_per_a = c / (a + epsilon_for_a)

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

    # bezier_pixel_intercepts(bezier_points)

    positive_bezier_points = bezier_points - bezier_points.min(axis=0, keepdims=True) + 1

    render_bezier(positive_bezier_points, np.ceil(positive_bezier_points.max(axis=0)).astype(np.uint32))
    pass


if __name__ == '__main__':
    main()
