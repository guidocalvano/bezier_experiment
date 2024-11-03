import numpy as np
import sys

from numpy.ma.core import negative


def roots_of_quadratic_old(a, b, c):
    is_line = a == 0.0
    negative_positive = np.reshape(np.array([-1, 1]), (1,) * len(a.shape) + (2,))
    solutions =(-b[..., np.newaxis] + negative_positive *  (np.sqrt(b**2 - 4*a*c)[..., np.newaxis])) / \
                (2 * a + is_line * sys.float_info.epsilon)[..., np.newaxis]

    return solutions


def roots_of_quadratic(a, b, c):
    # https://youtu.be/wTEYApUX-N8?si=itIEaO0nMGeNHvl2&t=416
    rooted_part = b*b - 4 * a * c
    has_solution = (rooted_part >= 0.0)[..., np.newaxis]

    negative_positive = np.reshape(np.array([-1, 1]), (1,) * len(a.shape) + (2,))

    denominator = (negative_positive * b[..., np.newaxis] + np.sqrt(rooted_part)[..., np.newaxis])

    solutions = ((-negative_positive * 2.0) * c[..., np.newaxis] /
                 (denominator + (np.logical_and(denominator == 0, a[..., np.newaxis] != 0)) * sys.float_info.epsilon))

    return solutions, np.logical_and(has_solution, np.isfinite(solutions))


def roots_to_abc(x1, x2):
    return np.array((1, -x1 - x2, x1*x2))


def test():
    a, b, c = roots_to_abc(2,5)

    # solutions = roots_of_quadratic(a, b, c)

    # x1 and x2 are retrieved correctly
    solutions2, is_solution2 = roots_of_quadratic(a, b, c)
    assert list(np.sort(solutions2)) == [2.0, 5.0]
    assert list(is_solution2[np.argsort(solutions2)]) == [True, True]

    # line intersecting y = 0 at x = -.5
    solutions3, is_solution3 = roots_of_quadratic(np.array([0]), np.array([6]), np.array([3]))
    assert list(np.sort(solutions3[0])) == [-0.5, np.inf]
    assert list(is_solution3[0][np.argsort(solutions3[0])]) == [True, False]

    # x*x = 0
    # should have two solutions, and currently does
    solutions4, is_solution4 = roots_of_quadratic(np.array([1]), np.array([0]), np.array([0]))
    assert set(list(solutions4[0])) == {-0, 0}
    assert list(is_solution4[0]) == [True, True]

    # line intersecting 0, 0
    # currently has two solutions of 0 and -0
    # but maybe should have a 0 and an infinite solution?
    solutions5, is_solution5 = roots_of_quadratic(np.array([0]), np.array([1]), np.array([0]))
    assert np.sort(solutions5[0])[0] == -0
    assert np.isnan(np.sort(solutions5[0])[1])

    assert list(is_solution5[0][np.argsort(solutions5[0])]) == [True, False]

    # no solutions, above y = 0
    solutions6, is_solution6 = roots_of_quadratic(np.array([1]), np.array([0]), np.array([1]))
    assert list(is_solution6[0]) == [False, False]

    pass


if __name__ == '__main__':
    test()