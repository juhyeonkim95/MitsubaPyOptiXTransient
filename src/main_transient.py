import sys
from utils.config_utils import *
from core.renderer_constants import process_config
from utils.image_utils import save_image, save_image_numpy
import numpy as np
from tqdm import tqdm
import os

if __name__ == "__main__":
	argument = sys.argv
	if len(argument) > 1:
		config_file = argument[1]
	else:
		config_file = "../configs_transient/bunny_0.4.json"

	config = load_config_recursive(config_file)

	
	#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	from pyoptix import Compiler
	Compiler.clean()
	Compiler.keep_device_function = False
	file_dir = os.path.dirname(os.path.abspath(__file__))
	Compiler.add_program_directory(file_dir)

	from core.renderer import Renderer
	renderer = Renderer()

	tx_xs = []
	tx_ys = []
	tx_zs = []
	rx_xs = []
	rx_ys = []
	rx_zs = []

	tmin = config.get("tmin")
	tmax = 0
	tres = 0

	with open("../transient_data/simulate_airsas.sh") as f:
		while True:
			# Get next line from file
			line = f.readline()
			splited_lines = line.split(" ")
			for l in splited_lines:
				kv = l.split("=")
				if len(kv) == 2:
					k = kv[0]
					v = kv[1]
					if k == "tx_x":
						tx_xs.append(float(v))
					elif k == "tx_y":
						tx_ys.append(float(v))
					elif k == "tx_z":
						tx_zs.append(float(v))
					elif k == "rx_x":
						rx_xs.append(float(v))
					elif k == "rx_y":
						rx_ys.append(float(v))
					elif k == "rx_z":
						rx_zs.append(float(v))
					elif k == "tMin":
						tmin = float(v)
					elif k == "tMax":
						tmax = float(v)
					elif k == "tRes":
						tres = float(v)
			if not line:
				break

	# if "cylinder" in config_file and "specular" not in config_file:
	# 	tmin = 0
	# 	tmax = 4

	tx_coords = np.stack((tx_xs, tx_ys, tx_zs), axis=-1)
	rx_coords = np.stack((rx_xs, rx_ys, rx_zs), axis=-1)
	nbin = round((tmax - tmin) / tres) // 5
	print("Samples %d, tmin: %f, tmax: %f, nbin: %d" % (tx_coords.shape[0], tmin, tmax, nbin))

	N = tx_coords.shape[0]
	histograms = np.zeros((N, nbin, 4), dtype=np.float32)
	transient_configs = {
		"transient_dist_max": tmax,
		"transient_dist_min": tmin,
		"transient_bin_num": nbin
	}

	renderer.init(**config, **transient_configs)

	for i in tqdm(range(N)):
		renderer.update_camera_and_emitter_position(rx_coords[i, :], tx_coords[i, :])
		result = renderer.render(**config, **transient_configs)
		transient_histogram = result["transient_histogram"]
		histograms[i, :, :] = transient_histogram

	np.save(config_file.replace(".json", ".npy"), histograms)


	if config.get("save_image", False):
		output_dir = config.get("output_dir", Path(config_file).parent)
		file_name = os.path.splitext(config_file)[0]
		file_name = file_name.split("/")[-1]
		file_name = config.get("file_name", file_name)
		file_path = os.path.join(output_dir, file_name)
		if config.get("convert_ldr", True):
			save_image(result["image"], file_path)
		else:
			save_image_numpy(result["image"], file_path)
