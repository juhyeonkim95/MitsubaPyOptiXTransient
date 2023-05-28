import copy

from pyoptix import Context, Buffer
from core.scene import Scene
import time
from core.utils.math_utils import *
import matplotlib.pyplot as plt

from core.renderer_constants import *
from utils.logging_utils import *
from utils.timing_utils import *
import gc
from core.optix_scene import OptiXSceneContext
from core.loader.loader_general import load_camera


class Renderer:
    def __init__(self, **kwargs):
        """
        This is created only once!
        :param scale:
        :param force_all_diffuse:
        """
        # Optix Context
        self.optix_context = None

        self.width = 0
        self.height = 0
        self.scale = kwargs.get("scale", 1)

        self.scene = None
        self.scene_name = None
        self.reference_image = None
        self.scene_octree = None
        self.context = None

        self.render_load_logger = load_logger('Render load logger')
        self.render_logger = load_logger('Render logger')
        self.render_logger.setLevel(logging.INFO)

    def init_scene_config(self, scene_name, scene_file_path=None):
        # load scene info (non optix)
        self.scene = Scene(scene_name)
        self.scene_name = scene_name

        if scene_file_path == None:
            scene_file_path = "../../scenes/%s/scene.xml" % scene_name

        self.scene.load_scene_from(scene_file_path)
        self.width = self.scene.width // self.scale
        self.height = self.scene.height // self.scale

    def reset_output_buffers(self, width, height):
        self.context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        
    def load_scene(self, scene_name, forced=False, scene_file_path=None):
        if self.scene_name != scene_name or forced:
            del self.optix_context
            del self.scene
            gc.collect()
            self.context = Context()

            with time_measure("[1] Optix Context Create", self.render_load_logger):
                self.optix_context = OptiXSceneContext(self.context)

            with time_measure("[2] Scene Config Load", self.render_load_logger):
                self.init_scene_config(scene_name, scene_file_path)

            with time_measure("[3] OptiX Load", self.render_load_logger):
                self.optix_context.load_scene(self.scene)
            return True
        else:
            self.render_load_logger.info("Skipped loading scene because it has been already loaded")
            return False

    def update_camera_and_emitter_position(self, camera_position, emitter_position):
        sensor_node_original = self.scene.sensor_node

        sensor_node_new = copy.deepcopy(sensor_node_original)
        transform_node = sensor_node_new.find('*[@name="toWorld"]')
        lookat_node = transform_node.find("lookat")
        lookat_node.set("origin", ",".join(map(str, camera_position.tolist())))
        new_camera = load_camera(sensor_node_new)
        self.scene.camera = new_camera
        self.optix_context.init_camera(self.scene)

        self.scene.light_list[0].position = emitter_position
        self.optix_context.load_scene_lights(self.scene)

    def init(
        self,
        scene_name="cornell-box",
        scene_file_path=None,
        spp=256,
        time_limit_in_sec=-1,
        show_picture=False,
        samples_per_pass=16,
        max_depth=8,
        rr_begin_depth=4,
        convert_ldr=True,
        print_all_result=False,
        skip_all_steps=False,
        **kwargs
    ):
        self.scale = kwargs.get("scale", 1)
        optix_created = self.load_scene(scene_name, scene_file_path=scene_file_path)
        if not optix_created:
            self.optix_context.update_program()

        # just for shorter name
        context = self.context
        scene = self.scene
        width = self.width
        height = self.height

        # Transient Related
        context['transient_dist_max'] = np.array(kwargs.get("transient_dist_max"), dtype=np.float32)
        context['transient_dist_min'] = np.array(kwargs.get("transient_dist_min"), dtype=np.float32)
        context['transient_bin_num'] = np.array(kwargs.get("transient_bin_num"), dtype=np.uint32)
        context['transient_radiance_histogram'] = Buffer.empty((kwargs.get("transient_bin_num"), max_depth), dtype=np.float32, buffer_type='o', drop_last_dim=False)

        # path tracing related
        context['rr_begin_depth'] = np.array(rr_begin_depth, dtype=np.uint32)
        context['max_depth'] = np.array(max_depth, dtype=np.uint32)

        self.reset_output_buffers(width, height)
        context.validate()
        context.compile()

    def render(
        self,
        scene_name="cornell-box",
        scene_file_path=None,
        spp=256,
        time_limit_in_sec=-1,
        show_picture=False,
        samples_per_pass=16,
        max_depth=8,
        rr_begin_depth=4,
        convert_ldr=True,
        print_all_result=False,
        skip_all_steps=False,
        **kwargs
    ):

        # just for shorter name
        context = self.context
        scene = self.scene
        width = self.width
        height = self.height
        self.reset_output_buffers(width, height)
        context['transient_radiance_histogram'] = Buffer.empty((kwargs.get("transient_bin_num"), max_depth), dtype=np.float32, buffer_type='o', drop_last_dim=False)

        current_samples_per_pass = samples_per_pass
        if samples_per_pass == -1:
            current_samples_per_pass = spp

        list_time_optix_launch = []
        left_samples = spp
        completed_samples = 0
        n_pass = 0

        '''
        Main Render Loop
        '''
        try:
            with timeout(time_limit_in_sec):
                while left_samples > 0:
                    context["samples_per_pass"] = np.array(current_samples_per_pass, dtype=np.uint32)
                    context["completed_sample_number"] = np.array(completed_samples, dtype=np.uint32)

                    # Run OptiX program
                    with record_elapsed_time("OptiX Launch", list_time_optix_launch, self.render_logger):
                        context.launch(0, width, height)

                    completed_samples += current_samples_per_pass

                    # update next pass
                    left_samples -= current_samples_per_pass
                    current_samples_per_pass = samples_per_pass
                    current_samples_per_pass = min(current_samples_per_pass, left_samples)
                    n_pass += 1

        except TimeoutError:
            self.render_logger.info("%f sec is over" % time_limit_in_sec)

        # histogram
        transient_signal_histogram = self.context['transient_radiance_histogram'].to_array()
        transient_signal_histogram /= (completed_samples * width * height)

        if show_picture:
            transient_dist_min = kwargs.get("transient_dist_min", 10)
            transient_dist_max = kwargs.get("transient_dist_max", 10)
            transient_bin_num = kwargs.get("transient_bin_num", 10)
            
            ts = np.linspace(transient_dist_min, transient_dist_max, transient_bin_num)

            # (1) Image
            final_raw_image = self.context['output_buffer'].to_array()
            final_raw_image = final_raw_image / completed_samples
            # final_raw_image = np.flipud(final_raw_image)
            hdr_image = final_raw_image[:, :, 0:3]
            ldr_image = LinearToSrgb(ToneMap(hdr_image, 1.5))
            final_image = ldr_image if convert_ldr else hdr_image
            plt.imshow(final_image)
            plt.show()

            # (2) Transient Signal
            plt.plot(ts, transient_signal_histogram[:, 1], label="1")
            plt.plot(ts, transient_signal_histogram[:, 2], label="2")
            plt.plot(ts, transient_signal_histogram[:, 3], label="3")
            plt.xlabel('time')
            plt.ylabel('intensity')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.legend(title="Bounce number")
            plt.show()

        # Summarize Result

        results = dict()
        results["transient_histogram"] = transient_signal_histogram

        return results
