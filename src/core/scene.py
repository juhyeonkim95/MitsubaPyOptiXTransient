from core.utils.math_utils import *
from pyoptix import GeometryInstance,  Transform, GeometryGroup, Acceleration
import os

from core.optix_mesh import OptixMesh
import xml.etree.ElementTree as ET
from utils.logging_utils import *
from utils.timing_utils import *
from utils.image_utils import *
from core.loader.loader_general import *
from core.shapes.objmesh import OBJMesh, InstancedShape, Shape
from itertools import chain
from core.textures.texture import *
from core.emitters.envmap import EnvironmentMap
from pathlib import Path


def add_transform(transformation, geometry_instance):
    if transformation is None:
        transformation = np.eye(4, dtype=np.float32)
    elif isinstance(transformation, dict):
        return add_animation(transformation, geometry_instance)

    geometry_instance['velocity'] = np.array([0, 0, 0], dtype=np.float32)

    gg = GeometryGroup(children=[geometry_instance])
    gg.set_acceleration(Acceleration("Trbvh"))

    transform = Transform(children=[gg])
    transform.set_matrix(False, transformation.transpose())
    transform.add_child(gg)
    return transform


def add_animation(animation, geometry_instance):
    matrices_full = []
    matrices = []
    times = []
    for keyframe_time, matrix in animation.items():
        matrices_full.append(matrix)
        matrices.append(matrix.transpose()[0:3, :])
        times.append(float(keyframe_time))

    p1 = matrices_full[0] * Vector3([0, 0, 0])
    p2 = matrices_full[1] * Vector3([0, 0, 0])
    velocity = (p1 - p2) / (times[0] - times[1])
    print("Velocity", velocity)

    geometry_instance['velocity'] = np.array(velocity, dtype=np.float32)

    gg = GeometryGroup(children=[geometry_instance])
    gg.set_acceleration(Acceleration("Trbvh"))

    transform = Transform(children=[gg])
    transform.set_motion_range(0, 1)
    transform.set_motion_border_mode("clamp", "clamp")
    print(matrices)
    transform.set_motion_keys(len(matrices), "matrix", [matrices[0], matrices[1]])

    transform.add_child(gg)
    return transform


class Scene:
    def __init__(self, name):
        self.name = name

        self.camera = None
        self.sensor_node = None

        # name list (need to be virtually loaded after)
        self.texture_name_list = []

        self.texture_name_to_optix_index_dictionary = {}
        self.texture_sampler_list = []

        self.material_id_to_index_dictionary = {}
        self.material_list = []
        self.shape_list = []
        self.texture_list = []
        self.light_list = []

        self.obj_name_list = []
        self.obj_geometry_dict = {}

        self.geometry_instances = []
        self.light_instances = []

        self.folder_path = None
        self.bbox = BoundingBox()

        self.width = 0
        self.height = 0
        self.has_envmap = False

    def load_scene_from(self, file_name):
        """
        Load scene from file.
        :param file_name: target file name
        :return:
        """
        doc = ET.parse(file_name)
        root = doc.getroot()
        include_file = load_value(root, "include_file", None)

        if include_file is not None:
            path = Path(file_name)
            parent_path = path.parent
            include_file = os.path.join(parent_path, include_file)
            include_tree = ET.parse(include_file)
            include_root = include_tree.getroot()
            for elem in include_root:
                if elem.tag == "default":
                    root.insert(0, elem)
                else:
                    root.append(elem)
            root.remove(root.find('*[@name="%s"]' % "include_file"))

        scene_load_logger = load_logger('Scene config loader')
        shape_load_logger = load_logger('Shape config loader')
        material_load_logger = load_logger('Material config loader')
        emitter_load_logger = load_logger('Emitter config loader')

        sensor = root.find("sensor")

        # 0. load scene image size.
        film = load_film(sensor.find("film"))
        self.height, self.width = film.height, film.width
        # print log
        scene_load_logger.info("0. Image size loaded")
        scene_load_logger.info("[Size] : %dx%d" % (self.width, self.height))

        # 1. load camera
        self.sensor_node = root.find("sensor")
        self.camera = load_camera(root.find("sensor"))

        # print log
        scene_load_logger.info("1. Camera Loaded")
        scene_load_logger.info(str(self.camera))

        # 2. load geometry + material
        self.load_shapes(root)
        # print log
        shape_load_logger.info("2. Shape Loaded")
        shape_load_logger.info("Total %d shapes" % len(self.shape_list))
        for shape in self.shape_list:
            shape_load_logger.info(str(shape))
            shape_load_logger.info("\t- material id : %s" % shape.bsdf.id)
        material_load_logger.info("3. Material Loaded")
        material_load_logger.info("Total %d materials" % len(self.material_list))
        for material in self.material_list:
            material_load_logger.info(str(material))

        self.folder_path = os.path.dirname(file_name)

        # 3. load independent emitter info
        for emitter_node in root.findall('emitter'):
            emitter = load_emitter(emitter_node)
            if isinstance(emitter, EnvironmentMap):
                self.has_envmap = True
            emitter.list_index = len(self.light_list)
            self.light_list.append(emitter)

        # print log
        emitter_load_logger.info("4. Emitter Loaded")
        emitter_load_logger.info("Total %d emitters" % len(self.light_list))
        for light in self.light_list:
            emitter_load_logger.info(str(light))

    def load_shapes(self, root):
        """
        Load shape information that includes material information, from root node
        :param root: root node
        :return:
        """
        shape_list = []
        obj_list = []
        anonymous_material_count = 0

        for node in root.findall('shape'):
            # 1. load shape
            shape = load_single_shape(node)

            # if shape includes obj mesh, this would be loaded after.
            if isinstance(shape, OBJMesh) and shape.obj_file_name not in obj_list:
                obj_list.append(shape.obj_file_name)

            # 2. load material
            material_ref = node.find("ref")

            # 2.1 defined by reference
            if material_ref is not None:
                bsdf_id = material_ref.attrib["id"]

                # 2.1.1 already loaded
                if bsdf_id in self.material_id_to_index_dictionary:
                    bsdf_index = self.material_id_to_index_dictionary[bsdf_id]
                    material = self.material_list[bsdf_index]

                # 2.2.2 not loaded
                else:
                    bsdf = root.find('bsdf[@id="%s"]' % bsdf_id)
                    material = self.load_new_material(bsdf)

            # 2.2 defined inside the node (anonymous material).
            else:
                bsdf_id = "anonymous_material_%d" % anonymous_material_count
                anonymous_material_count += 1
                bsdf = node.find("bsdf")
                material = self.load_new_material(bsdf, bsdf_id=bsdf_id)

            # 3. load emitter
            if node.find("emitter") is not None:
                from core.emitters.area import AreaLight
                emitter = load_emitter(node.find("emitter"))
                assert isinstance(emitter, AreaLight)
                shape.emitter = emitter
                emitter.shape = shape
                emitter.list_index = len(self.light_list)
                self.light_list.append(emitter)

                # print(shape.emitter)
                # radiance = load_value(emitter, "radiance", default=np.array([1, 1, 1], dtype=np.float32))
                # material_light = MaterialParameter("light_%d" % light_count)
                # light_count += 1
                # material_light.type = 'light'
                # material_light.is_double_sided = material.is_double_sided
                # material_light.emission = radiance
                # material = material_light

            shape.bsdf = material
            shape_list.append(shape)

        self.shape_list = shape_list
        self.obj_name_list = obj_list

    def load_new_material(self, bsdf, bsdf_id=None):
        """
        Load new material and save to material list
        :param bsdf: bsdf node
        :param bsdf_id: given bsdf name
        :return:
        """
        # load material information
        material = load_bsdf(bsdf)
        if material.id is None:
            material.id = bsdf_id

        # append material to material list
        self.material_id_to_index_dictionary[material.id] = len(self.material_list)
        material.list_index = len(self.material_list)
        self.material_list.append(material)

        return material

    def optix_load_objs(self, program_dictionary):
        """
        Load OBJ files and store it as OptiX Geometry instance
        :param program_dictionary:
        :return:
        """
        mesh_bb = program_dictionary['tri_mesh_bb']
        mesh_it = program_dictionary['tri_mesh_it']

        for obj_file_name in self.obj_name_list:
            mesh = OptixMesh(mesh_bb, mesh_it)
            mesh.load_from_file(self.folder_path + "/" + obj_file_name)
            self.obj_geometry_dict[obj_file_name] = mesh

    def optix_load_textures(self):
        """
        Load texture data and store it as OptiX object.
        :return:
        """
        from core.textures.bitmap import BitmapTexture

        # environment map
        for light in self.light_list:
            if isinstance(light, EnvironmentMap):
                self.has_envmap = True
                tex_sampler = load_texture_sampler(self.folder_path, light.filename, gamma=1)
                self.texture_sampler_list.append(tex_sampler)
                light.envmapID = tex_sampler.get_id()
                print("ENV loaded", light.envmapID, light.filename)

        # get all materials
        self.texture_list = []
        for material in self.material_list:
            self.texture_list += material.get_textures()
        self.texture_name_list = []
        for texture in self.texture_list:
            if isinstance(texture, BitmapTexture) and texture.filename not in self.texture_name_list:
                self.texture_name_list.append(texture.filename)

        print("Load texture list")
        print(self.texture_name_list)

        for texture_name in self.texture_name_list:
            tex_sampler = load_texture_sampler(self.folder_path, texture_name, gamma=2.2)
            self.texture_name_to_optix_index_dictionary[texture_name] = tex_sampler.get_id()
            self.texture_sampler_list.append(tex_sampler)

        # assign optix id and list id to texture
        for (i, texture) in enumerate(self.texture_list):
            if isinstance(texture, BitmapTexture):
                texture.texture_optix_id = self.texture_name_to_optix_index_dictionary[texture.filename]
            texture.list_index = i

    def optix_create_geometry_instances(self, program_dictionary, material_dict, force_all_diffuse=False):
        Shape.program_dictionary = program_dictionary

        opaque_material = material_dict['opaque_material']
        cutout_material = material_dict['cutout_material']
        light_material = material_dict['light_material']

        geometry_instances = []
        light_instances = []

        for shape in self.shape_list:
            shape_type = shape.shape_type
            geometry = shape.to_optix_geometry()
            bbox = shape.get_bbox()

            # (1) create geometry
            if shape_type == "obj":
                mesh = self.obj_geometry_dict[shape.obj_file_name]
                shape.mesh = mesh
                geometry = mesh.geometry
                bbox = mesh.bbox
            elif shape_type == "rectangle":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()
            elif shape_type == "sphere":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()
            elif shape_type == "disk":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()
            elif shape_type == "cube":
                geometry = shape.to_optix_geometry()
                bbox = shape.get_bbox()

            # (2) create material
            if shape.emitter is not None:
                target_material = light_material
            else:
                bsdf_type = shape.bsdf.bsdf_type
                if bsdf_type == "mask":
                    target_material = cutout_material
                else:
                    target_material = opaque_material

            geometry_instance = GeometryInstance(geometry, target_material)
            mat_id = np.array(shape.bsdf.list_index, dtype=np.int32)
            emitter_id = np.array(shape.emitter.list_index if shape.emitter is not None else -1, dtype=np.int32)
            bsdf_type = np.array(int(shape.bsdf.optix_bsdf_type), dtype=np.int32)

            geometry_instance['materialId'] = mat_id
            geometry_instance["lightId"] = emitter_id
            geometry_instance['programId'] = bsdf_type

            if isinstance(shape, InstancedShape):
                if isinstance(shape.transform, Matrix44):
                    bbox = get_bbox_transformed(bbox, np.array(shape.transform.transpose(), dtype=np.float32))
                else:
                    merged_bbox = None
                    for keyframe_time, transform in shape.transform.items():
                        ith_frame_bbox = get_bbox_transformed(bbox, np.array(transform.transpose(), dtype=np.float32))
                        if merged_bbox is None:
                            merged_bbox = ith_frame_bbox
                        else:
                            merged_bbox = get_bbox_merged(merged_bbox, ith_frame_bbox)
                    bbox = merged_bbox

            # merge bbox
            self.bbox = get_bbox_merged(self.bbox, bbox)

            if isinstance(shape, OBJMesh):
                geometry_instance["faceNormals"] = np.array(1 if shape.face_normals else 0, dtype=np.int32)

            if isinstance(shape, InstancedShape):
                transform = add_transform(shape.transform, geometry_instance)
            else:
                transform = add_transform(None, geometry_instance)
            #if shape.transformation is not None:
            # geometry_instance["transformation"] = shape.transformation
            if target_material == light_material:
                light_instances.append(transform)
            else:
                geometry_instances.append(transform)

        self.geometry_instances = geometry_instances
        self.light_instances = light_instances
