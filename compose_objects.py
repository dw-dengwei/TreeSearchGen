import bpy
import os
import tempfile
from tqdm import tqdm
import sys
import math
from mathutils import Vector
import json
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
    output_path = os.path.join(output_dir, 'top_view_retextured.png')
    scene.render.filepath = output_path

    # 渲染顶视图
    bpy.ops.render.render(write_still=True)
    print(f"Top view rendered and saved at: {output_path}")
    
    return cam_obj

def render_side_view(output_dir, room_dimension, distance=8, angle=45):
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
    output_path = os.path.join(output_dir, f'side_view_{angle}_retextured.png')
    scene.render.filepath = output_path

    # 渲染侧视图
    bpy.ops.render.render(write_still=True)
    print(f"Side view rendered and saved at: {output_path}")
    
    return cam_obj

scene_dir = sys.argv[-1]

glb_path = os.path.join(scene_dir, '16_scene.glb')
decompose_folder = os.path.join(scene_dir, 'decompose')
export_glb_path = os.path.join(scene_dir, 'recombined_scene.glb')
texture_folder_relative = os.path.join(scene_dir, 'decompose', 'new_texture')  # 新纹理相对路径
object_orientation_path = os.path.join(scene_dir, '15_object_orientation.json')
object_orientation = json.load(open(object_orientation_path))

# 启用GPU渲染
enable_gpu_rendering([0])  # 使用第一个GPU设备

# 删除所有现有对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

set_transparent_background()

# 创建一个临时文件夹用于存放修改后的 MTL
temp_dir = tempfile.mkdtemp()

# 处理每个OBJ
for file in os.listdir(decompose_folder):
    if file.endswith(".obj"):
        base_name = os.path.splitext(file)[0]
        obj_path = os.path.join(texture_folder_relative, base_name.replace('_', '#'), 'txt_stage2', 'tile_res_0', 'mesh.obj')

        # mtl_path = os.path.join(decompose_folder, f"{base_name}.mtl")

        # temp_obj_path = obj_path  # 默认用原来的OBJ
        # temp_mtl_path = mtl_path  # 默认用原来的MTL

        # # --- 如果存在MTL ---
        # if os.path.exists(mtl_path):
        #     # 在内存中改写
        #     new_lines = []
        #     with open(mtl_path, 'r') as f:
        #         for line in f:
        #             if line.startswith('map_Kd'):
        #                 parts = line.strip().split()
        #                 if len(parts) >= 2:
                            # new_texture_path = os.path.join(texture_folder_relative, base_name.replace('_', "#"), 'txt_stage2/tile_res_0/albedo.png')
        #                     new_texture_path = os.path.abspath(new_texture_path)
        #                     new_line = f"map_Kd {new_texture_path}\n"
        #                     new_lines.append(new_line)

        #                     normal_path = os.path.abspath(os.path.join(decompose_folder, 'textures', parts[1].replace('albedo', 'normal')))
        #                     new_normal_line = f'map_Bump {normal_path}\n'
        #                     new_lines.append(new_normal_line)
        #                 else:
        #                     new_lines.append(line)
        #             elif line.startswith('map_Ke'):
        #                 parts = line.strip().split()
        #                 emission_path = os.path.abspath(os.path.join(decompose_folder, parts[1]))
        #                 new_line = f"map_Ke {emission_path}\n"
        #                 new_lines.append(new_line)
        #             else:
        #                 new_lines.append(line)

        #     # 写到临时目录
        #     temp_mtl_path = os.path.join(temp_dir, f"{base_name}.mtl")
        #     with open(temp_mtl_path, 'w') as f:
        #         f.writelines(new_lines)

        #     # 需要让OBJ文件知道去用这个新的MTL，怎么办？
        #     # 复制OBJ到临时目录，同时修改mtllib指向新的MTL
        #     temp_obj_path = os.path.join(temp_dir, f"{base_name}.obj")
        #     with open(obj_path, 'r') as f_in, open(temp_obj_path, 'w') as f_out:
        #         for line in f_in:
        #             if line.startswith('mtllib'):
        #                 f_out.write(f"mtllib {os.path.basename(temp_mtl_path)}\n")
        #             else:
        #                 f_out.write(line)

        # --- 导入处理后的OBJ ---
        bpy.ops.import_scene.obj(filepath=obj_path)
        
        # 获取刚导入的物体
        imported_obj = bpy.context.selected_objects[0]
        
        ori_to_ang = [180, 90, 0, 270]
        # 在Z轴上旋转
        rot = None
        loc = None
        size = None
        for name, ori in object_orientation.items():
            if name.replace('_', ' ').replace('#', ' ') == base_name.replace('_', ' ').replace('#', ' '):
                rot = ori['ori']
                loc = ori['loc']
                size = ori['size']
                print(f"Rotating {base_name} to {ori_to_ang[ori['ori']]} degrees")
                break
        if rot is None:
            rot = 0
            print(f"Warning: {base_name} not found in object_orientation.json")
        
        # 确保物体被选中
        bpy.ops.object.select_all(action='DESELECT')
        imported_obj.select_set(True)
        bpy.context.view_layer.objects.active = imported_obj
        
        # 1. 计算物体的底部几何中心点（变换前）
        bbox_corners = [imported_obj.matrix_world @ Vector(corner) for corner in imported_obj.bound_box]
        min_z = min(corner.z for corner in bbox_corners)
        bottom_center = Vector((
            (max(corner.x for corner in bbox_corners) + min(corner.x for corner in bbox_corners)) / 2,
            (max(corner.y for corner in bbox_corners) + min(corner.y for corner in bbox_corners)) / 2,
            min_z
        ))
        
        # 2. 设置3D游标位置为底部几何中心点，作为变换的基准点
        bpy.context.scene.cursor.location = bottom_center
        bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'
        
        # 3. 缩放操作
        if size is not None:
            if rot in [1, 3]:
                size = [size[1], size[0], size[2]]
            # 计算当前尺寸
            current_dimensions = (
                max(corner.x for corner in bbox_corners) - min(corner.x for corner in bbox_corners),
                max(corner.y for corner in bbox_corners) - min(corner.y for corner in bbox_corners),
                max(corner.z for corner in bbox_corners) - min(corner.z for corner in bbox_corners)
            )

            print(current_dimensions, size)
            scale_z_origin = sum([
                size[0] / current_dimensions[0],
                size[1] / current_dimensions[1],
            ]) / 2

            scale_z_target = size[2] / current_dimensions[2]

            if scale_z_origin > scale_z_target * 2:
                scale_z = scale_z_target
            else:
                scale_z = scale_z_origin
            
            # 计算缩放比例
            scale_factors = (
                size[0] / current_dimensions[0],
                size[1] / current_dimensions[1],
                scale_z
                # target_dimensions[2] / current_dimensions[2]
            )
            
            # 应用缩放
            imported_obj.scale = Vector(scale_factors)
            
            # 应用缩放变换，使其生效
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # 4. 旋转操作
        if rot is not None:
            # 将角度转换为弧度
            rotation_angle = math.radians(ori_to_ang[rot] + 180)
            # 绕Z轴旋转
            bpy.ops.transform.rotate(value=rotation_angle, orient_axis='Z')
            # 应用旋转变换，使其生效
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        
        # 5. 平移操作
        if loc is not None:
            # 重新计算变换后的底部几何中心点
            bbox_corners = [imported_obj.matrix_world @ Vector(corner) for corner in imported_obj.bound_box]
            min_z = min(corner.z for corner in bbox_corners)
            current_bottom_center = Vector((
                (max(corner.x for corner in bbox_corners) + min(corner.x for corner in bbox_corners)) / 2,
                (max(corner.y for corner in bbox_corners) + min(corner.y for corner in bbox_corners)) / 2,
                min_z
            ))
            
            # 计算需要的位移量（目标位置 - 当前位置）
            target_location = Vector(loc)
            translation = target_location - current_bottom_center
            
            # 应用平移
            imported_obj.location = imported_obj.location + translation


# 选中所有物体
bpy.ops.object.select_all(action='SELECT')

# 计算场景尺寸
scene_objects = bpy.context.selected_objects
if scene_objects:
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for obj in scene_objects:
        for v in obj.bound_box:
            world_v = obj.matrix_world @ Vector(v)
            min_x = min(min_x, world_v.x)
            max_x = max(max_x, world_v.x)
            min_y = min(min_y, world_v.y)
            max_y = max(max_y, world_v.y)
            min_z = min(min_z, world_v.z)
            max_z = max(max_z, world_v.z)
    
    room_dimension = [max_x - min_x, max_y - min_y, max_z - min_z]
else:
    room_dimension = [10, 10, 3]  # 默认房间尺寸

# 渲染顶视图和侧视图
render_top_view(scene_dir, room_dimension)
render_side_view(scene_dir, room_dimension, angle=45, distance=room_dimension[0])

# 导出为GLB
bpy.ops.export_scene.gltf(
    filepath=export_glb_path,
    export_format='GLB',
    use_selection=False,
    export_texcoords=True,
    export_normals=True,
    export_materials='EXPORT',
    export_cameras=True,  # 导出相机
    export_lights=True,   # 导出灯光
)
