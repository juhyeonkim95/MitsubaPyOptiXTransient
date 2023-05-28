import copy

from pyoptix import Context, Buffer
from core.scene import Scene
import time
from core.utils.math_utils import *
import matplotlib.pyplot as plt

from core.renderer_constants import *
from path_guiding.radiance_record import QTable
from utils.logging_utils import *
from utils.timing_utils import *
import gc
from core.optix_scene import OptiXSceneContext, update_optix_configs
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
        self.context['hit_count_buffer'] = Buffer.empty((height, width, 1), dtype=np.uint32, buffer_type='o', drop_last_dim=True)
        self.context['path_length_buffer'] = Buffer.empty((height, width, 1), dtype=np.uint32, buffer_type='o', drop_last_dim=True)
        self.context['scatter_type_buffer'] = Buffer.empty((height, width, 2), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        self.context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
        self.context['output_buffer2'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o',  drop_last_dim=True)

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

    def render(
            self,
            scene_name="cornell-box",
            scene_file_path=None,
            spp=256,
            time_limit_in_sec=-1,
            time_limit_init_ignore_step=0,
            time_consider_only_optix=False,
            sampling_strategy=SAMPLE_BRDF,
            q_table_update_method=Q_UPDATE_MONTE_CARLO,
            show_picture=False,
            samples_per_pass=16,
            learning_method="incremental",
            accumulative_q_table_update=True,
            max_depth=8,
            rr_begin_depth=4,
            directional_mapping_method="cylindrical",
            use_bsdf_first_force=True,
            force_update_q_table=False,
            bsdf_sampling_fraction=0.5,
            min_epsilon=0.0,
            no_exploration=False,
            convert_ldr=True,
            use_next_event_estimation=False,
            print_all_result=False,
            skip_all_steps=False,
            **kwargs
    ):
        if not skip_all_steps:
            self.scale = kwargs.get("scale", 1)
            sampling_strategy = key_value_to_int("sampling_strategy", sampling_strategy)
            q_table_update_method = key_value_to_int("q_table_update_method", q_table_update_method)

            # load scene info & init optix
            update_optix_configs(
                sampling_strategy=sampling_strategy,
                q_table_update_method=q_table_update_method,
                spatial_data_structure_type=kwargs.get("spatial_data_structure_type", "grid"),
                directional_data_structure_type=kwargs.get("directional_data_structure_type", "grid"),
                directional_mapping_method=directional_mapping_method,
                use_next_event_estimation=use_next_event_estimation
            )

            optix_created = self.load_scene(scene_name, scene_file_path=scene_file_path)
            if not optix_created:
                self.optix_context.update_program()

            # just for shorter name
            context = self.context
            scene = self.scene
            width = self.width
            height = self.height

            context['transient_dist_max'] = np.array(kwargs.get("transient_dist_max"), dtype=np.float32)
            context['transient_dist_min'] = np.array(kwargs.get("transient_dist_min"), dtype=np.float32)
            context['transient_bin_num'] = np.array(kwargs.get("transient_bin_num"), dtype=np.uint32)
            context['transient_radiance_histogram'] = Buffer.empty((kwargs.get("transient_bin_num"), max_depth), dtype=np.float32, buffer_type='o', drop_last_dim=False)

            # rendering related
            context['rr_begin_depth'] = np.array(rr_begin_depth, dtype=np.uint32)
            context['max_depth'] = np.array(max_depth, dtype=np.uint32)
            context["bsdf_sampling_fraction"] = np.array(bsdf_sampling_fraction, dtype=np.float32)
            context["sampling_strategy"] = np.array(sampling_strategy, dtype=np.uint32)
            context["q_table_update_method"] = np.array(q_table_update_method, dtype=np.uint32)
            context["accumulative_q_table_update"] = np.array(1 if accumulative_q_table_update else 0, dtype=np.uint32)

            need_q_table_update = force_update_q_table or not ((sampling_strategy == SAMPLE_UNIFORM) or (sampling_strategy == SAMPLE_BRDF))

            context["need_q_table_update"] = np.array(1 if need_q_table_update else 0, dtype=np.uint32)

            room_size = scene.bbox.bbox_max - scene.bbox.bbox_min
            context['scene_bbox_min'] = scene.bbox.bbox_min
            context['scene_bbox_max'] = scene.bbox.bbox_max
            context['scene_bbox_extent'] = room_size

            self.reset_output_buffers(width, height)
            QTable.register_empty_context(context)

            q_table = None
            if need_q_table_update:
                q_table = QTable(
                    directional_mapping_method=directional_mapping_method,
                    accumulative_q_table_update=accumulative_q_table_update, **kwargs
                )
                q_table.register_to_context(context)

            hit_sum = 0
            list_hit_counts = []
            output_images = []

            is_budget_time = time_limit_in_sec > 0

            context.validate()
            context.compile()


        else:
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
        final_histogram = self.context['transient_radiance_histogram'].to_array()
        final_histogram /= (completed_samples * width * height)

        if show_picture:
            transient_dist_min = kwargs.get("transient_dist_min", 10)
            transient_dist_max = kwargs.get("transient_dist_max", 10)
            transient_bin_num = kwargs.get("transient_bin_num", 10)
            print(transient_dist_min)
            print(transient_dist_max)
            print(transient_bin_num)
            
            ts = np.linspace(transient_dist_min, transient_dist_max, transient_bin_num)

            # (0) Image
            final_raw_image = self.context['output_buffer'].to_array()
            final_raw_image = final_raw_image / completed_samples
            final_raw_image = np.flipud(final_raw_image)
            hdr_image = final_raw_image[:, :, 0:3]
            ldr_image = LinearToSrgb(ToneMap(hdr_image, 1.5))
            final_image = ldr_image if convert_ldr else hdr_image

            plt.imshow(final_image)
            plt.show()
            print(ts.shape, "TS SHAPE!!")

            # plt.plot(final_histogram[:, 0], label="1")
            plt.plot(ts, final_histogram[:, 1], label="1")
            plt.plot(ts, final_histogram[:, 2], label="2")
            plt.plot(ts, final_histogram[:, 3], label="3")
            plt.ylabel('')
            plt.legend()
            plt.show()

        # Summarize Result

        results = dict()
        results["transient_histogram"] = final_histogram

        return results
