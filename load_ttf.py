import numpy as np
import matplotlib.pyplot as plt
from main import render_bezier, bezier_pixel_intercepts

def main():
    file_path = './JetBrainsMono-Regular.ttf'

    raw_content = np.fromfile(file_path, dtype=np.uint8)

    scaler_type = raw_content[0:4].view(np.dtype('>u4'))[0]
    num_tables = raw_content[4:6].view(np.dtype('>u2'))[0]
    search_range = raw_content[6:8].view(np.dtype('>u2'))[0]
    entry_selector = raw_content[8:10].view(np.dtype('>u2'))[0]
    range_shift = raw_content[10:12].view(np.dtype('>u2'))[0]

    TABLE_DIRECTORY_OFFSET = 12
    TABLE_DIRECTORY_ELEMENT_COUNT = 4
    TABLE_DIRECTORY_ELEMENT_SIZE = 4
    TABLE_DIRECTORY_BYTE_COUNT = TABLE_DIRECTORY_ELEMENT_COUNT * TABLE_DIRECTORY_ELEMENT_SIZE * num_tables
    TABLE_DIRECTORY_END = TABLE_DIRECTORY_OFFSET + TABLE_DIRECTORY_BYTE_COUNT

    table_data = np.reshape(raw_content[TABLE_DIRECTORY_OFFSET:TABLE_DIRECTORY_END].view(np.dtype('>u4')), (num_tables, TABLE_DIRECTORY_ELEMENT_COUNT))

    table_tag = table_data[:, 0].copy().view(np.dtype('S4'))
    table_checksum = table_data[:, 1].copy().view(np.dtype('>u4'))
    table_offset = table_data[:, 2].copy().view(np.dtype('>u4'))
    table_length = table_data[:, 3].copy().view(np.dtype('>u4'))

    table_tag_dict = {tag.decode('ascii'): index for index, tag in enumerate(table_tag)}
    glyf_index = table_tag_dict['glyf']
    glyf_offset = table_offset[glyf_index]
    glyf_length = table_length[glyf_index]
    glyf_end = glyf_offset + glyf_length
    glyf_table_raw = raw_content[glyf_offset:glyf_end]

    NUMBER_OF_CONTOURS_OFFSET = 0
    BOUNDING_BOX_OFFSET = 2
    END_POINT_OF_CONTOUR_OFFSET = BOUNDING_BOX_OFFSET + 4*2

    number_of_contours = glyf_table_raw[NUMBER_OF_CONTOURS_OFFSET:2].view('>i2')[0]

    instruction_length_offset = (END_POINT_OF_CONTOUR_OFFSET + 2 * np.abs(number_of_contours))
    x_min, y_min, x_max, y_max = glyf_table_raw[BOUNDING_BOX_OFFSET:END_POINT_OF_CONTOUR_OFFSET].view('>i2')

    end_points_of_contours = glyf_table_raw[END_POINT_OF_CONTOUR_OFFSET:instruction_length_offset].view('>i2')

    instruction_length = glyf_table_raw[instruction_length_offset:instruction_length_offset + 2].view('>i2')[0]
    instruction_offset = instruction_length_offset + 2
    instruction_end = instruction_offset + instruction_length

    instructions = glyf_table_raw[instruction_offset:instruction_end].view('u1')

    flags_offset = instruction_end

    point_count = end_points_of_contours[-1] + 1
    flags = np.empty([point_count], dtype=np.uint8)

    processed_flag_count = 0
    flag_index = flags_offset

    while processed_flag_count < point_count:  # I hate file formats that force using loops...
        next_flag = glyf_table_raw[flag_index:(flag_index + 1)].view(np.uint8)[0]
        flags[processed_flag_count] = next_flag
        if not(next_flag & 0b00001000):
            processed_flag_count += 1
            flag_index += 1
            continue
        # if repeat flag is set
        repeat_count = glyf_table_raw[(flag_index + 1):(flag_index + 2)].view(np.uint8)[0]
        flags[processed_flag_count:(processed_flag_count + repeat_count)] = next_flag
        processed_flag_count += repeat_count
        flag_index += 2

    is_on_curve = (flags & 0b0000_0001) > 0

    x_byte_count = 2 - ((flags & 0b00000010) > 0).astype(np.uint8)
    y_byte_count = 2 - ((flags & 0b00000100) > 0).astype(np.uint8)

    must_skip_x = np.logical_and(x_byte_count == 2, (flags & 0b0001_0000) > 0)
    must_skip_y = np.logical_and(y_byte_count == 2, (flags & 0b0010_0000) > 0)
    must_invert_sign_x = np.logical_and(x_byte_count == 1, (flags & 0b0001_0000) == 0)
    must_invert_sign_y = np.logical_and(y_byte_count == 1, (flags & 0b0010_0000) == 0)
    sign_factor_x = 1 - 2 * must_invert_sign_x
    sign_factor_y = 1 - 2 * must_invert_sign_y

    x_byte_count_with_skips = x_byte_count * (1 - must_skip_x)
    x_index = np.cumsum(x_byte_count_with_skips) - x_byte_count_with_skips  # exclusive scan
    x_offset = flag_index + x_index

    x_coordinate_diffs = np.zeros([flags.shape[0]])
    single_byte_mask_x = x_byte_count == 1
    single_byte_indices_x = x_index[single_byte_mask_x]
    two_byte_mask_x = x_byte_count_with_skips == 2
    two_byte_indices_x = x_index[two_byte_mask_x]
    x_coordinate_diffs[single_byte_mask_x] = sign_factor_x[single_byte_mask_x] * glyf_table_raw[x_offset[single_byte_mask_x]].view(np.dtype('>u1'))
    x_coordinate_diffs[two_byte_mask_x] = np.array([glyf_table_raw[x_offset[two_byte_mask_x]], glyf_table_raw[x_offset[two_byte_mask_x] + 1]]).T.copy().view(np.dtype('>i2'))[:, 0]

    # X TESTED UNTIL HERE, Y IS PROBABLY WRONG, BECAUSE Y COORDINATES START AFTER X, AND I AM NOT SURE IF THEY ARE CORRECTED FOR THE OFFSET AFTER THE X COORDINATES

    y_byte_count_with_skips = y_byte_count * (1 - must_skip_y)
    y_index = np.cumsum(y_byte_count_with_skips) - y_byte_count_with_skips  # exclusive scan
    y_offset = flag_index + np.sum(x_byte_count_with_skips) + y_index

    y_coordinate_diffs = np.zeros([flags.shape[0]])
    single_byte_mask_y = y_byte_count == 1
    single_byte_indices_y = y_index[single_byte_mask_y]
    two_byte_mask_y = y_byte_count_with_skips == 2
    two_byte_indices_y = y_index[two_byte_mask_y]
    y_coordinate_diffs[single_byte_mask_y] = sign_factor_y[single_byte_mask_y] * glyf_table_raw[
        y_offset[single_byte_mask_y]].view(np.dtype('>u1'))
    y_coordinate_diffs[two_byte_mask_y] = np.array(
        [glyf_table_raw[y_offset[two_byte_mask_y]], glyf_table_raw[y_offset[two_byte_mask_y] + 1]]).T.copy().view(
        np.dtype('>i2'))[:, 0]

    x_coordinate = np.cumsum(x_coordinate_diffs)
    y_coordinate = np.cumsum(y_coordinate_diffs)

    plt.scatter(x_coordinate, y_coordinate)
    ax = plt.gca()
    plt.title("stored points")

    ax.set_aspect('equal', adjustable='box')
    plt.show()

    x_coordinate_offset = flag_index
    current_offset = x_coordinate_offset

    #@TODO COMPOSITE GLYPHS SHOULD HAVE A CORRECTION HERE, segmented  roll...

    start_points_of_contours = np.roll((end_points_of_contours + 1) % is_on_curve.shape[0], 1)
    source_contour_end = np.zeros(is_on_curve.shape, dtype=np.int32)
    source_contour_end[end_points_of_contours] = 1
    source_contour_start = np.roll(source_contour_end, 1, axis=0)

    source_contour_index = np.cumsum(source_contour_end) - source_contour_end
    source_contour_size = end_points_of_contours - start_points_of_contours + 1

    #@TODO: refactor this into a segmented roll
    next_source_contour_index = (np.arange(is_on_curve.shape[0]) - start_points_of_contours[source_contour_index] + 1) % source_contour_size[source_contour_index] + start_points_of_contours[source_contour_index]
    must_insert_midpoint = is_on_curve == is_on_curve[next_source_contour_index]
    insert_count = np.sum(must_insert_midpoint)

    source_original_index = np.arange(is_on_curve.shape[0])
    move_original_size = np.cumsum(must_insert_midpoint) - must_insert_midpoint
    target_original_index = source_original_index + move_original_size

    start_points_of_target_contours = target_original_index[start_points_of_contours]
    end_points_of_target_contours = target_original_index[end_points_of_contours] + must_insert_midpoint[end_points_of_contours]
    target_contour_count = end_points_of_target_contours - start_points_of_target_contours + 1

    element_count = x_coordinate.shape[0] + insert_count

    target_contour_end = np.zeros(element_count, dtype=np.int32)
    target_contour_end[end_points_of_target_contours] = 1

    target_contour_start = np.zeros(element_count, dtype=np.int32)
    target_contour_start[start_points_of_target_contours] = 1

    target_contour_index = np.cumsum(target_contour_end) - target_contour_end

    x_coordinate_with_inserts = np.empty(element_count)
    y_coordinate_with_inserts = np.empty(element_count)

    mean_a_index = target_original_index[must_insert_midpoint]
    target_insert_index = (mean_a_index + 1) # % x_coordinate_with_inserts.shape[0]

    mean_b_index = (target_insert_index + 1 - start_points_of_target_contours[target_contour_index[mean_a_index]]) % target_contour_count[target_contour_index[mean_a_index]] + start_points_of_target_contours[target_contour_index[mean_a_index]]

    x_coordinate_with_inserts[target_original_index] = x_coordinate[source_original_index]
    y_coordinate_with_inserts[target_original_index] = y_coordinate[source_original_index]

    x_coordinate_with_inserts[target_insert_index] = (x_coordinate_with_inserts[mean_a_index] + x_coordinate_with_inserts[mean_b_index]) / 2.0
    y_coordinate_with_inserts[target_insert_index] = (y_coordinate_with_inserts[mean_a_index] + y_coordinate_with_inserts[mean_b_index]) / 2.0

    plt.scatter(x_coordinate_with_inserts, y_coordinate_with_inserts)
    ax = plt.gca()
    plt.title("with inserts")
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    plt.plot(x_coordinate_with_inserts, y_coordinate_with_inserts)
    ax = plt.gca()
    plt.title('with inserts as lines')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    element_count_with_looped = element_count + number_of_contours
    x_coordinate_with_looped_contours = np.empty(element_count_with_looped)
    y_coordinate_with_looped_contours = np.empty(element_count_with_looped)

    x_coordinate_with_looped_contours[np.arange(element_count) + target_contour_index] = x_coordinate_with_inserts
    y_coordinate_with_looped_contours[np.arange(element_count) + target_contour_index] = y_coordinate_with_inserts

    endpoint_of_looped_target_contours = np.arange(number_of_contours) + 1 + end_points_of_target_contours
    startpoint_of_looped_target_contours = np.roll((endpoint_of_looped_target_contours + 1) % x_coordinate_with_looped_contours.shape[0], 1)
    x_coordinate_with_looped_contours[endpoint_of_looped_target_contours] = x_coordinate_with_looped_contours[startpoint_of_looped_target_contours]  # x_coordinate_with_inserts[start_points_of_target_contours]
    y_coordinate_with_looped_contours[endpoint_of_looped_target_contours] = y_coordinate_with_looped_contours[startpoint_of_looped_target_contours]# y_coordinate_with_inserts[start_points_of_target_contours]

    plt.plot(x_coordinate_with_looped_contours, y_coordinate_with_looped_contours)
    plt.scatter(x_coordinate_with_looped_contours, y_coordinate_with_looped_contours)

    ax = plt.gca()
    plt.title('with looped contours as lines')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    plt.plot(x_coordinate_with_looped_contours[startpoint_of_looped_target_contours[0]:endpoint_of_looped_target_contours[0] + 1],
             y_coordinate_with_looped_contours[startpoint_of_looped_target_contours[0]:endpoint_of_looped_target_contours[0] + 1])
    plt.scatter(x_coordinate_with_looped_contours[:endpoint_of_looped_target_contours[0]], y_coordinate_with_looped_contours[:endpoint_of_looped_target_contours[0]])
    plt.scatter(x_coordinate_with_looped_contours[endpoint_of_looped_target_contours[0]], y_coordinate_with_looped_contours[endpoint_of_looped_target_contours[0]], c="red", s=4)
    plt.scatter(x_coordinate_with_looped_contours[0], y_coordinate_with_looped_contours[0], c="green", s = 2)


    ax = plt.gca()
    plt.title('first looped contour')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    plt.plot(x_coordinate_with_looped_contours[startpoint_of_looped_target_contours[1]:endpoint_of_looped_target_contours[1] + 1],
             y_coordinate_with_looped_contours[startpoint_of_looped_target_contours[1]:endpoint_of_looped_target_contours[1] + 1])
    plt.scatter(x_coordinate_with_looped_contours[endpoint_of_looped_target_contours[0]:endpoint_of_looped_target_contours[1]], y_coordinate_with_looped_contours[endpoint_of_looped_target_contours[0]:endpoint_of_looped_target_contours[1]])
    plt.scatter(x_coordinate_with_looped_contours[endpoint_of_looped_target_contours[0]], y_coordinate_with_looped_contours[endpoint_of_looped_target_contours[0]], c="red")
    plt.scatter(x_coordinate_with_looped_contours[endpoint_of_looped_target_contours[1]], y_coordinate_with_looped_contours[endpoint_of_looped_target_contours[1]], c="green")

    ax = plt.gca()
    plt.title('second looped contour')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    plt.plot(x_coordinate_with_looped_contours[startpoint_of_looped_target_contours[2]:endpoint_of_looped_target_contours[2] + 1],
             y_coordinate_with_looped_contours[startpoint_of_looped_target_contours[2]:endpoint_of_looped_target_contours[2] + 1])
    plt.scatter(x_coordinate_with_looped_contours[endpoint_of_looped_target_contours[1]:endpoint_of_looped_target_contours[2]], y_coordinate_with_looped_contours[endpoint_of_looped_target_contours[1]:endpoint_of_looped_target_contours[2]])
    plt.scatter(x_coordinate_with_looped_contours[endpoint_of_looped_target_contours[1]], y_coordinate_with_looped_contours[endpoint_of_looped_target_contours[1]], c="red")
    plt.scatter(x_coordinate_with_looped_contours[endpoint_of_looped_target_contours[2]], y_coordinate_with_looped_contours[endpoint_of_looped_target_contours[2]], c="green")

    ax = plt.gca()
    plt.title('third looped contour')
    ax.set_aspect('equal', adjustable='box')
    plt.show()




    for i in range(3):
        bezier_pixel_intercepts(np.array([x_coordinate_with_looped_contours[startpoint_of_looped_target_contours[i]:endpoint_of_looped_target_contours[i] + 1],
                                      y_coordinate_with_looped_contours[startpoint_of_looped_target_contours[i]:endpoint_of_looped_target_contours[i] + 1]]).T / 30)

    for i in range(3):
        render_bezier(np.array([x_coordinate_with_looped_contours[
                                          startpoint_of_looped_target_contours[i]:endpoint_of_looped_target_contours[
                                                                                      i] + 1],
                                          y_coordinate_with_looped_contours[
                                          startpoint_of_looped_target_contours[i]:endpoint_of_looped_target_contours[
                                                                                      i] + 1]]).T / 30)

    # messed up serial format

    pass


if __name__ == '__main__':
    main()