import sys
from utils.config_utils import *
from core.renderer_constants import process_config
from utils.image_utils import save_image, save_image_numpy


if __name__ == "__main__":
	argument = sys.argv
	if len(argument) > 1:
		config_file = argument[1]
	else:
		#config_file = "../configs_tof/cornell-box/point-light/max_depth_2/homodyne/analytic.json"
		#config_file = "../configs_tof/simple-cube-static/point-light/max_depth_2/heterodyne/full_mc_uniform.json"
		config_file = "../configs_transient/brdf.json"

	config = load_config_recursive(config_file)

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	from pyoptix import Compiler
	Compiler.clean()
	Compiler.keep_device_function = False
	file_dir = os.path.dirname(os.path.abspath(__file__))
	Compiler.add_program_directory(file_dir)

	from core.renderer import Renderer
	renderer = Renderer()
	result = renderer.render(**config)

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
