import os


def tof_option_to_folder(**kwargs):
    force_collocated_point_light = kwargs.get("force_collocated_point_light", True)
    max_depth = kwargs.get("max_depth", 2)
    homodyne = kwargs.get("homodyne", True)

    option_point_light = "point_light" if force_collocated_point_light else "original_light"
    option_max_bounce = "max_depth_%d" % max_depth
    option_homodyne = "homo" if homodyne else "hetero"
    output_folder = os.path.join(option_point_light, option_max_bounce, option_homodyne)
    return output_folder
