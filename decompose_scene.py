import bpy
import os
import sys
import math
import json


scene_dir = sys.argv[-1]

glb_path = os.path.join(scene_dir, '16_scene.glb')
object_orientation_path = os.path.join(scene_dir, '15_object_orientation.json')
object_orientation = json.load(open(object_orientation_path))

if not os.path.exists(glb_path):
    print(f'File not found: {glb_path}')
    exit()
export_folder = os.path.join(scene_dir, 'decompose')
temp_texture_folder = os.path.join(export_folder, "textures")

# 删除所有对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 导入GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

# 确保导出目录存在
os.makedirs(export_folder, exist_ok=True)
os.makedirs(temp_texture_folder, exist_ok=True)

ori_to_ang = [180, 90, 0, 270]

# 1. 解包材质中的贴图图像
for image in bpy.data.images:
    if image.packed_file:  # 如果是内嵌的
        image_name = bpy.path.clean_name(image.name)
        save_path = os.path.join(temp_texture_folder, f"{image_name}.png")
        image.filepath_raw = save_path
        image.file_format = 'PNG'
        image.save()

# # 2. 遍历每个对象单独导出
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # 在Z轴上旋转90度（π/2弧度）
        rot = None
        for name, ori in object_orientation.items():
            if name == obj.name:
                rot = ori['ori']
                rot = -ori_to_ang[rot] + 180
                print(f"Rotating {obj.name} to {rot} degrees")
                break
        if rot is None:
            rot = 0
            print(f"Warning: {obj.name} not found in object_orientation.json")

        if rot is not None:
            bpy.ops.transform.rotate(value=rot / 360 * 2 * math.pi, orient_axis='Z')

        obj_name = bpy.path.clean_name(obj.name)
        obj_export_path = os.path.join(export_folder, f"{obj_name}.obj")

        bpy.ops.export_scene.obj(
            filepath=obj_export_path,
            use_selection=True,
            use_materials=True,
            path_mode='COPY',  # 这时候COPY就能正常拷贝刚刚保存的图片了
            axis_forward='-Z',
            axis_up='Y'
        )

        print(f"导出完成: {obj_export_path}")