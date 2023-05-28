import sys
from utils.config_utils import *
from core.renderer_constants import process_config
from utils.image_utils import save_image, save_image_numpy
import numpy as np
from tqdm import tqdm
import os

if __name__ == "__main__":
	argument = sys.argv
	config_file = argument[1]

	config = load_config_recursive(config_file)

	from pyoptix import Compiler
	Compiler.clean()
	Compiler.keep_device_function = False
	file_dir = os.path.dirname(os.path.abspath(__file__))
	Compiler.add_program_directory(file_dir)

	from core.renderer import Renderer
	renderer = Renderer()

	transient_configs = {
		"transient_dist_max": config.get("tMax", 1),
		"transient_dist_min": config.get("tMin", 0),
		"transient_bin_num": config.get("nBin", 10000),
	}

	rx_x = config.get("rx_x", 0.0)
	rx_y = config.get("rx_y", 0.0)
	rx_z = config.get("rx_z", 0.0)
	tx_x = config.get("tx_x", 0.0)
	tx_y = config.get("tx_y", 0.0)
	tx_z = config.get("tx_z", 0.0)
	rx_coord = np.array([rx_x, rx_y, rx_z], dtype=float)
	tx_coord = np.array([tx_x, tx_y, tx_z], dtype=float)
	

	renderer.init(**config, **transient_configs)
	renderer.update_camera_and_emitter_position(rx_coord, tx_coord)
	
	result = renderer.render(**config, **transient_configs)
	transient_histogram = result["transient_histogram"]
	
	output_file_name = config.get("output_file_name")

	np.save(output_file_name, transient_histogram)