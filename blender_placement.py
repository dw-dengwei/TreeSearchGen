import bpy
import sys
import os
import numpy as np
import math
import argparse
# from utils.objaverse import OBJATHOR_ASSETS_DIR
# import objaverse
# import compress_json
PROJECT_ROOT = '/home/dw/code/llm-scene'
sys.path.append(PROJECT_ROOT)
import compress_pickle
import compress_json
from mathutils import Vector, Matrix
from utils.load_json import load_json
import json


OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser("~/.objathor-assets")
)

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
HD_BASE_VERSION = os.environ.get("HD_BASE_VERSION", "2023_09_23")
HOLODECK_BASE_DATA_DIR = os.path.join(
    OBJATHOR_ASSETS_BASE_DIR, "holodeck", HD_BASE_VERSION
)
OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")
HOLODECK_THOR_FEATURES_DIR = os.path.join(HOLODECK_BASE_DATA_DIR, "thor_object_data")
HOLODECK_THOR_ANNOTATIONS_PATH = os.path.join(
    HOLODECK_BASE_DATA_DIR, "thor_object_data", "annotations.json.gz"
)

if ASSETS_VERSION > "2023_09_23":
    THOR_COMMIT_ID = "8524eadda94df0ab2dbb2ef5a577e4d37c712897"
else:
    THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"

def enable_gpu_rendering(cuda_ids):
    """
    启用 GPU 渲染模式并优化渲染设置
    """
    # 设置渲染引擎为 Cycles
    bpy.context.scene.render.engine = 'CYCLES'

    # 确保选择 GPU 作为设备
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences

    # 设置渲染设备为 GPU
    cycles_prefs.compute_device_type = 'CUDA'  # 如果使用 NVIDIA GPU
    
    # 激活所有可用的 GPU 设备
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = False
    
    # 启用指定的 GPU 设备
    for index in cuda_ids:
        if index < len(cycles_prefs.devices):
            device = cycles_prefs.devices[index]
            if 'CUDA' in device.type:
                device.use = True
                print(f"Enabled GPU device: {device.name}")
            else:
                print(f"Device at index {index} is not a CUDA device.")
        else:
            print(f"Invalid GPU index: {index}")

    # 将 Cycles 设置为使用 GPU 设备
    bpy.context.scene.cycles.device = 'GPU'
    print("GPU rendering enabled.")

# 场景创建和渲染功能保持不变...

objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
database = {**objathor_annotations}

def render_top_view(output_dir, room_dimension, height=10.0):
    """
    渲染场景的顶视图
    :param output_dir: 输出图像的目录
    :param room_dimension: 房间尺寸 [width, length, height]
    :param height: 相机高度
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene

    # 设置渲染参数
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.image_settings.file_format = 'PNG'

    # 获取场景的中心点
    scene_center = Vector((room_dimension[0] / 2, room_dimension[1] / 2, 0))

    # 添加一个强光源在相机位置，确保场景有足够的照明
    light_data = bpy.data.lights.new(name="TopLight", type='SUN')
    light_data.energy = 5.0  # 增加光源强度
    light_obj = bpy.data.objects.new(name="TopLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (scene_center.x, scene_center.y, height - 1)
    light_obj.rotation_euler = (math.radians(0), 0, 0)  # 光线朝下

    # 设置相机在正上方，并指向场景中心
    cam_location = (scene_center.x, scene_center.y, height)
    cam = bpy.data.cameras.new(name='TopCamera')
    cam_obj = bpy.data.objects.new('TopCamera', cam)
    bpy.context.collection.objects.link(cam_obj)
    
    cam_obj.location = Vector(cam_location)
    # 旋转相机使其朝下
    cam_obj.rotation_euler = (math.radians(0), 0, 0)
    
    # 设置相机为正交模式，以获得更好的顶视图效果
    cam.type = 'ORTHO'
    # 设置正交相机的缩放比例，确保能够看到整个房间
    cam.ortho_scale = max(room_dimension[0], room_dimension[1]) * 1.1
    
    # 将相机设置为活动相机
    scene.camera = cam_obj

    # 设置输出文件路径
    output_path = os.path.join(output_dir, 'top_view.png')
    scene.render.filepath = output_path

    # 渲染顶视图
    bpy.ops.render.render(write_still=True)
    print(f"Top view rendered and saved at: {output_path}")
    
    return cam_obj

def render_side_view(output_dir, room_dimension, distance=8, angle=0):
    """
    渲染场景的侧视图
    :param output_dir: 输出图像的目录
    :param room_dimension: 房间尺寸 [width, length, height]
    :param distance: 相机与场景中心的距离
    :param angle: 相机水平角度（以度为单位），0表示从X轴正方向看，90表示从Y轴正方向看
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene

    # 设置渲染参数
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.image_settings.file_format = 'PNG'

    # 获取场景的中心点
    scene_center = Vector((room_dimension[0] / 2, room_dimension[1] / 2, room_dimension[2] / 2 if len(room_dimension) > 2 else 1.5))

    # 添加一个强光源在相机位置，确保场景有足够的照明
    light_data = bpy.data.lights.new(name="SideLight", type='SUN')
    light_data.energy = 5.0  # 增加光源强度
    light_obj = bpy.data.objects.new(name="SideLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    
    # 计算相机在水平面上的位置
    radians = math.radians(angle)
    cam_x = scene_center.x + distance * math.cos(radians)
    cam_y = scene_center.y + distance * math.sin(radians)
    cam_z = scene_center.z + 5 

    # 设置光源位置在相机附近
    light_obj.location = (cam_x, cam_y, cam_z + 2)
    # 光源朝向场景中心
    direction = light_obj.location - scene_center
    light_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

    # 设置相机位置和朝向
    cam = bpy.data.cameras.new(name='SideCamera')
    cam_obj = bpy.data.objects.new('SideCamera', cam)
    bpy.context.collection.objects.link(cam_obj)
    
    cam_obj.location = Vector((cam_x, cam_y, cam_z))
    # 计算相机朝向场景中心的旋转
    direction = cam_obj.location - scene_center
    cam_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    
    # 设置相机为透视模式，以获得更自然的侧视图效果
    cam.type = 'PERSP'
    # 设置相机视场角度
    cam.angle = math.radians(60)
    
    # 将相机设置为活动相机
    scene.camera = cam_obj

    # 设置输出文件路径
    output_path = os.path.join(output_dir, f'side_view_{angle}.png')
    scene.render.filepath = output_path

    # 渲染侧视图
    bpy.ops.render.render(write_still=True)
    print(f"Side view rendered and saved at: {output_path}")
    
    return cam_obj

def create_area_light(name, size, power, location):
    """
    创建一个面光源（Area Light）并设置其位置和强度。
    :param name: 光源名称
    :param size: 光源尺寸
    :param power: 光源强度（瓦特）
    :param location: 光源的位置 (x, y, z)
    """
    # 创建一个平面对象
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    light_obj = bpy.context.active_object
    light_obj.name = name

    # 将平面转换为发光材质
    light_material = bpy.data.materials.new(name="Light_Material")
    light_material.use_nodes = True
    emission_node = light_material.node_tree.nodes.new(type='ShaderNodeEmission')
    emission_node.inputs['Strength'].default_value = power
    emission_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # 白色光源

    # 将发光节点连接到材质输出
    material_output = light_material.node_tree.nodes.get('Material Output')
    light_material.node_tree.links.new(emission_node.outputs['Emission'], material_output.inputs['Surface'])

    # 将材质赋给平面对象
    light_obj.data.materials.append(light_material)

    # 旋转平面使其向下照射
    # light_obj.rotation_euler = (math.radians(90), 0, 0)

    return light_obj


def setup_camera(location, target):
    """
    创建一个相机并将其放置在指定位置，指向目标位置。
    :param location: 相机的位置 (x, y, z)
    :param target: 相机观测的目标位置 (x, y, z)
    """
    # 创建相机数据
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = Vector(location)

    # 计算朝向目标的旋转
    direction = cam_obj.location - Vector(target)
    cam_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

    # 将相机设置为活动相机
    bpy.context.scene.camera = cam_obj

    return cam_obj

def render_images(output_dir, angles, distance, height, room_dimension):
    """
    渲染不同视角的图像。
    :param output_dir: 输出图像的目录
    :param angles: 各个视角的水平角度列表（以度为单位）
    :param distances: 相机与场景中心的距离
    :param height: 相机的高度
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene

    # 设置渲染参数
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.image_settings.file_format = 'PNG'

    # 获取场景的中心点
    scene_center = Vector((room_dimension[0] / 2, room_dimension[1] / 2, 0))

    cam_seq = []
    for angle in angles:
        # 计算相机在水平面上的位置
        radians = math.radians(angle)
        cam_x = scene_center.x + distance * math.cos(radians)
        cam_y = scene_center.y + distance * math.sin(radians)
        cam_z = height

        # 设置相机在斜上方，并指向场景中心
        cam = setup_camera((cam_x, cam_y, cam_z), scene_center)
        cam_seq.append(cam)

        # 设置输出文件路径
        output_path = os.path.join(output_dir, f'render_{angle}.png')
        scene.render.filepath = output_path

        # 渲染当前视角
        bpy.ops.render.render(write_still=True)
        print(f"Rendered image saved at: {output_path}")

    return cam_seq

def create_camera_animation(cam_seq, total_frames):
    """
    根据 cam_seq 创建相机动画。
    :param cam_seq: 相机对象的列表
    :param total_frames: 动画总帧数
    """
    if not cam_seq:
        print("No cameras in sequence.")
        return

    # 获取场景中的相机
    scene = bpy.context.scene
    camera = bpy.context.scene.camera

    frames_per_cam = total_frames // len(cam_seq)

    for index, cam in enumerate(cam_seq):
        # 计算当前相机的帧范围
        start_frame = index * frames_per_cam + 1
        end_frame = start_frame + frames_per_cam - 1

        # 设置相机位置
        camera.location = cam.location
        camera.rotation_euler = cam.rotation_euler

        # 插入位置和旋转的关键帧
        camera.keyframe_insert(data_path="location", frame=start_frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=start_frame)

        if index < len(cam_seq) - 1:
            next_cam = cam_seq[index + 1]
            camera.location = next_cam.location
            camera.rotation_euler = next_cam.rotation_euler

            # 插入下一个位置和旋转的关键帧
            camera.keyframe_insert(data_path="location", frame=end_frame)
            camera.keyframe_insert(data_path="rotation_euler", frame=end_frame)

def render_animation(output_path, start_frame, end_frame, fps=24):
    """
    渲染相机动画并导出为视频文件。
    :param output_path: 输出视频的路径
    :param start_frame: 动画起始帧
    :param end_frame: 动画结束帧
    :param fps: 每秒帧数
    """
    scene = bpy.context.scene

    # 设置渲染参数
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    scene.render.fps = fps
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.render.filepath = output_path

    # 渲染动画
    bpy.ops.render.render(animation=True)

def rotate_vertices(vertices, angle, axis='Z'):
    # 计算旋转矩阵
    rotation_matrix = Matrix.Rotation(angle, 4, axis)
    # 应用旋转矩阵到每个顶点
    rotated_vertices = [rotation_matrix @ Vector(v) for v in vertices]
    return rotated_vertices

# 2. 创建 Blender 对象
def create_mesh(name, vertices, triangles, uvs):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    bpy.context.collection.objects.link(obj)
    
    # 3. 设置网格数据
    mesh.from_pydata(vertices, [], triangles)
    if uvs is not None:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        mesh.uv_layers.active = uv_layer

        # 遍历所有三角形并为每个顶点设置 UV 坐标
        for loop_index, loop in enumerate(mesh.loops):
            vertex_index = loop.vertex_index  # 获取当前循环对应的顶点索引
            uv_layer.data[loop_index].uv = (uvs[vertex_index][0], uvs[vertex_index][1])

    mesh.update()
    
    return obj

# 4. 设置材质
def create_material(name, albedoTexturePath, normalTexturePath, emissionTexturePath):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")

    # 设置 albedo
    if albedoTexturePath:
        albedo_tex = material.node_tree.nodes.new('ShaderNodeTexImage')
        albedo_tex.image = bpy.data.images.load(albedoTexturePath)
        material.node_tree.links.new(albedo_tex.outputs[0], bsdf.inputs[0])

    # 设置 normal
    if normalTexturePath:
        normal_tex = material.node_tree.nodes.new('ShaderNodeTexImage')
        normal_tex.image = bpy.data.images.load(normalTexturePath)
        normal_map = material.node_tree.nodes.new('ShaderNodeNormalMap')
        material.node_tree.links.new(normal_tex.outputs[0], normal_map.inputs[1])
        material.node_tree.links.new(normal_map.outputs[0], bsdf.inputs[17])

    # 设置 emission
    if emissionTexturePath:
        emission_tex = material.node_tree.nodes.new('ShaderNodeTexImage')
        emission_tex.image = bpy.data.images.load(emissionTexturePath)
        emission_bsdf = material.node_tree.nodes.new('ShaderNodeEmission')
        material.node_tree.links.new(emission_tex.outputs[0], emission_bsdf.inputs[0])
        material.node_tree.links.new(emission_bsdf.outputs[0], bsdf.inputs[18])
    
    return material


def set_object_dimensions(obj, target_dimensions):
    # 计算当前物体的尺寸
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    current_dimensions = (
        max(corner.x for corner in bbox_corners) - min(corner.x for corner in bbox_corners),
        max(corner.y for corner in bbox_corners) - min(corner.y for corner in bbox_corners),
        max(corner.z for corner in bbox_corners) - min(corner.z for corner in bbox_corners)
    )

    scale_z_origin = sum([
        target_dimensions[0] / current_dimensions[0],
        target_dimensions[1] / current_dimensions[1],
    ]) / 2

    scale_z_target = target_dimensions[2] / current_dimensions[2]

    if scale_z_origin > scale_z_target * 2:
        scale_z = scale_z_target
    else:
        scale_z = scale_z_origin
    
    # 计算缩放比例
    scale_factors = (
        target_dimensions[0] / current_dimensions[0],
        target_dimensions[1] / current_dimensions[1],
        scale_z
        # target_dimensions[2] / current_dimensions[2]
    )
    
    # 应用缩放
    obj.scale = Vector(scale_factors)
    return [
        target_dimensions[0], target_dimensions[1], current_dimensions[2] * scale_z
    ]


def create_object(uid, name, location, target_size, angle):
    obj_path = os.path.join(OBJATHOR_ASSETS_DIR, uid, f"{uid}.pkl.gz")
    data = compress_pickle.load(obj_path)
    albedo_texture_name = '/'.join(data['albedoTexturePath'].split('/')[-2:])
    albedo_texture_path = os.path.join(OBJATHOR_ASSETS_DIR, albedo_texture_name)
    normal_texture_name = '/'.join(data['normalTexturePath'].split('/')[-2:])
    normal_texture_path = os.path.join(OBJATHOR_ASSETS_DIR, normal_texture_name)
    emission_texture_name= '/'.join(data['emissionTexturePath'].split('/')[-2:])
    emission_texture_path = os.path.join(OBJATHOR_ASSETS_DIR, emission_texture_name)

    yRotOffset = (data['yRotOffset'] + angle) / 360 * 2 * math.pi  # z轴旋转

    vertices = np.array([[p['x'], -p['z'], p['y']] for p in data['vertices']])
    vertices = rotate_vertices(vertices, yRotOffset, 'Z')
    faces = np.array(data['triangles'])
    faces.resize(int(faces.shape[0] / 3), 3)

    uvs = np.array([[p['x'], p['y']] for p in data['uvs']])
    # 5. 创建对象和应用材质
    mesh_object = create_mesh(name, vertices, faces, uvs)
    material = create_material(name + "_Material", albedo_texture_path, normal_texture_path, emission_texture_path)
    if material:
        if len(mesh_object.data.materials) == 0:
            mesh_object.data.materials.append(material)
        else:
            mesh_object.data.materials[0] = material

    current_dimensions = set_object_dimensions(mesh_object, target_size)

    # 8. 计算底部中心点
    bbox_corners = [mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box]
    bottom_center = Vector((sum(corner.x for corner in bbox_corners) / 8,
                            sum(corner.y for corner in bbox_corners) / 8,
                            min(corner.z for corner in bbox_corners)))

    # 9. 定义目标位置
    target_location = Vector(location)  # 例如，您可以设置目标位置

    # 10. 计算新的位置
    new_location = target_location + (mesh_object.location - bottom_center)

    # 11. 移动物体
    mesh_object.location = new_location
    return current_dimensions


def create_scene(assets_path, furniture_layout_path, small_object_layout_path):
    assets_json = load_json(assets_path)
    object_orientation = {}
    assets = {}
    for a in assets_json['objects']:
        name = list(a.keys())[0]
        assets[name] = a[name]

    furniture_layout = load_json(furniture_layout_path)
    if os.path.exists(small_object_layout_path):
        small_object_layout = load_json(small_object_layout_path)
    room_dimension = furniture_layout['room_dimension']
    objects = []
    for area in furniture_layout['areas']:
        for fur in area['object_list']:
            fur['supported'] = 'floor'
            fur['location'].append(0)
            fur['uid'] = assets[fur['name']]['uid']
            objects.append(fur)

    if os.path.exists(small_object_layout_path):
        for fur in small_object_layout['areas']:
            for fur_name, fur_info in fur.items():
                for obj in fur_info['vis_furnitures_list']:
                    obj['supported'] = fur_name
                    obj['location'].append(0)
                    obj['uid'] = assets[fur_name + '_' + obj['name']]['uid']
                    objects.append(obj)


    def order(fur):
        if 'supported' not in fur:
            return 0
        if fur['supported'] == 'floor':
            return 0
        else:
            return 1

    dimensions = {}
    objects.sort(key=lambda f: order(f))
    print(objects)
    for fur in objects:
        uid = fur['uid']
        name = fur['name']
        location = fur['location']
        size = fur['size']
        if 'supported' not in fur:
            fur['supported'] = 'floor'
        supported = fur['supported']
        if supported != 'floor' and not any([s in supported for s in ['table', 'desk', 'stand', 'cabinet']]):
            continue
        if supported != 'floor':
            location[2] = dimensions[supported][2]
        ori_to_ang = [180, 90, 0, 270]
        angle = ori_to_ang[fur['orientation']]
        object_orientation[name] = {'ori': fur['orientation'], 'loc': location, 'size': size, 'uid': uid}
        current_dimension = create_object(uid, name, location, size, angle)
        if supported == 'floor':
            dimensions[name] = current_dimension

    create_floor(room_dimension)
    object_orientation['Floor'] = {'ori': 0, 'loc': [room_dimension[0] / 2, room_dimension[1] / 2, 0], 'size': [room_dimension[0], room_dimension[1], 0.02], 'uid': 'floor'}
    json.dump(object_orientation, open(os.path.join(project_dir, '15_object_orientation.json'), 'w'), indent=2)

def create_floor(room_dimension):
    """
    创建一个地板（扁平立方体），设置其长宽和高度，使用简单的图像纹理。
    :param room_dimension: 房间尺寸 [x, y]，其中：
        x: 地板的长度（X 方向）
        y: 地板的宽度（Y 方向）
    """
    x, y = room_dimension[0], room_dimension[1]
    thickness = 0.02  # 设置地板厚度为2厘米
    
    # 添加一个立方体对象
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x / 2, y / 2, thickness / 2))
    floor = bpy.context.active_object
    floor.name = 'Floor'

    # 设置立方体的缩放以达到指定的长宽和厚度
    floor.scale.x = x
    floor.scale.y = y
    floor.scale.z = thickness

    # 创建UV映射
    bpy.context.view_layer.objects.active = floor
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 创建一个新的图像
    image_size = 1024
    image = bpy.data.images.new(name="floor_texture", width=image_size, height=image_size)
    
    # 创建木地板纹理图案
    pixels = []
    for y in range(image_size):
        for x in range(image_size):
            # 创建木纹图案
            wood_pattern = (((x // 64) + (y // 64)) % 2) * 0.3 + 0.4
            # 添加一些随机噪声
            noise = ((x * y) % 100) / 500.0
            color_value = wood_pattern + noise
            # 使用棕色色调
            r = color_value * 0.8
            g = color_value * 0.5
            b = color_value * 0.3
            pixels.extend([r, g, b, 1.0])
    
    # 更新图像像素
    image.pixels = pixels
    image.update()
    image.pack()  # 确保图像被打包到.blend文件中

    # 设置地板的材质
    floor_material = bpy.data.materials.new(name="Floor_Material")
    floor_material.use_nodes = True
    nodes = floor_material.node_tree.nodes
    links = floor_material.node_tree.links
    
    # 清除所有现有节点
    nodes.clear()
    
    # 创建基础节点
    output = nodes.new('ShaderNodeOutputMaterial')
    principled_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    
    # 设置图像纹理
    tex_image.image = image
    
    # 设置节点位置
    output.location = (300, 0)
    principled_bsdf.location = (0, 0)
    tex_image.location = (-300, 0)
    tex_coord.location = (-700, 0)
    mapping.location = (-500, 0)
    
    # 连接节点
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
    links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Base Color'])
    links.new(principled_bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # 设置UV缩放
    mapping.inputs['Scale'].default_value[0] = 4.0
    mapping.inputs['Scale'].default_value[1] = 4.0
    
    # 设置材质属性
    principled_bsdf.inputs['Roughness'].default_value = 0.7
    principled_bsdf.inputs['Specular'].default_value = 0.3
    
    floor.data.materials.append(floor_material)

    # 调整UV缩放以匹配房间尺寸
    for face in floor.data.polygons:
        for loop_idx in face.loop_indices:
            uv = floor.data.uv_layers.active.data[loop_idx]
            uv.uv[0] *= x
            uv.uv[1] *= y

    return floor

def suppress_render_output():
    """
    禁用渲染过程中显示的详细信息。
    """
    # 设置 Blender 的调试级别为 0
    bpy.app.debug = False
    bpy.app.debug_value = 0

    # # 重定向标准输出和标准错误到 Null
    # sys.stdout = open(os.devnull, 'w')
    # sys.stderr = open(os.devnull, 'w')

enable_gpu_rendering([0])

suppress_render_output()

def set_transparent_background():
    """
    设置渲染背景为透明。
    """
    scene = bpy.context.scene

    # 设置渲染引擎为 Cycles（确保当前使用的是 Cycles）
    scene.render.engine = 'CYCLES'

    # 启用透明背景
    scene.render.film_transparent = True

    # 确保图像格式支持 Alpha 通道（PNG 格式）
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'  # 启用 Alpha 通道

for obj in bpy.data.objects:
    obj.select_set(True)
bpy.ops.object.delete()
# 调用函数以启用透明背景
set_transparent_background()

# 处理Blender命令行参数
# Blender特殊处理：在命令行中，所有"--"后的参数才会传递给脚本
# 例如: blender --background --python script.py -- --project_dir=/path --write_glb
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

# 使用argparse处理命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', type=str, required=True, help='project directory')
parser.add_argument('--write_glb', action='store_true', help='export glb file or not')
args = parser.parse_args(argv)

project_dir = args.project_dir
write_glb = args.write_glb

assets_path = os.path.join(
    project_dir, '12_retrieved_results_with_style.json'
)

furniture_layout_path = os.path.join(
    project_dir, '13_furniture_layout.json'
)

small_object_layout_path = os.path.join(
    project_dir, '14_small_object_layout.json'
)

try:
    create_scene(assets_path, furniture_layout_path, small_object_layout_path)
except Exception as e:
    print(f'Error: {assets_path}')
    raise e

# 渲染顶视图并保存
project_dir = os.path.dirname(assets_path)
layout = load_json(assets_path)
room_dimension = layout['room_dimension']
render_top_view(project_dir, room_dimension)
# 添加侧视图渲染，从不同角度拍摄
render_side_view(project_dir, room_dimension, angle=45, distance=room_dimension[0])
# render_images(project_dir, [0], 10, 10, room_dimension)

if write_glb:
    bpy.ops.export_scene.gltf(filepath=os.path.join(project_dir, '16_scene.glb'), export_format='GLB', export_cameras=True)

    