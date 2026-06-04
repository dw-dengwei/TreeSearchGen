import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from io import BytesIO
from PIL import Image
from utils.logger import logger
import matplotlib.patches as patches
import matplotlib.image as mpimg
from random import shuffle
from emoji_list import Emojis


# 创建长方体的顶点列表
def create_box(size, position):
    w, d, h = size
    half_w, half_d = w / 2, d / 2
    x_w, y_d, z_h = position
    return [
        # Bottom face
        [[x_w - half_w, y_d - half_d, z_h], 
         [x_w - half_w, y_d + half_d, z_h], 
         [x_w + half_w, y_d + half_d, z_h], 
         [x_w + half_w, y_d - half_d, z_h]],
        # Top face
        [[x_w - half_w, y_d - half_d, z_h + h],
         [x_w - half_w, y_d + half_d, z_h + h],
         [x_w + half_w, y_d + half_d, z_h + h],
         [x_w + half_w, y_d - half_d, z_h + h]],
        # Front face
        [[x_w - half_w, y_d + half_d, z_h],
         [x_w - half_w, y_d + half_d, z_h + h],
         [x_w + half_w, y_d + half_d, z_h + h],
         [x_w + half_w, y_d + half_d, z_h]],
        # Back face
        [[x_w - half_w, y_d - half_d, z_h],
         [x_w - half_w, y_d - half_d, z_h + h],
         [x_w + half_w, y_d - half_d, z_h + h],
         [x_w + half_w, y_d - half_d, z_h]],
        # Left face
        [[x_w - half_w, y_d - half_d, z_h],
         [x_w - half_w, y_d - half_d, z_h + h],
         [x_w - half_w, y_d + half_d, z_h + h],
         [x_w - half_w, y_d + half_d, z_h]],
        # Right face
        [[x_w + half_w, y_d - half_d, z_h],
         [x_w + half_w, y_d - half_d, z_h + h],
         [x_w + half_w, y_d + half_d, z_h + h],
         [x_w + half_w, y_d + half_d, z_h]],
    ]

# 可视化三维物体
def visualize_view(objects, ax, elev, azim):
    ax.view_init(elev=elev, azim=azim)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    g_min_x = g_min_y = g_min_z = float('inf')
    g_max_x = g_max_y = g_max_z = float('-inf')
    for obj in objects:
        size = obj['size']  # (length, width, height)
        position = obj['location']  # (x, y, z)
        category = obj['name']
        o_type = obj['type']
        
        # 获取物体的顶点
        box_faces = create_box(size, position)
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        for face in box_faces:
            for vertex in face:
                x, y, z = vertex
                # 更新最小值和最大值
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                min_z = min(min_z, z)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                max_z = max(max_z, z)
        g_min_x = min(g_min_x, min_x)
        g_min_y = min(g_min_y, min_y)
        g_min_z = min(g_min_z, min_z)
        g_max_x = max(g_max_x, max_x)
        g_max_y = max(g_max_y, max_y)
        g_max_z = max(g_max_z, max_z)
        
        # facecolors = 'cyan' if o_type == 'anchor' else 'red'
        color = 'white'
        e_color = 'black'
        if o_type == 'anchor':
            color = 'cyan'
            e_color = 'cyan'
        if o_type == 'target':
            color = 'red'
            e_color = 'red'
        # 创建3D多边形
        ax.add_collection3d(Poly3DCollection(box_faces, facecolors=color, linewidths=1, edgecolors=e_color, alpha=0.60))
        
        # 在物体上方标注类别
        ax.text(position[0], position[1], position[2] + size[2], category, color=color)

        if elev == 90 and azim == -90:
            orientation = None
            if 'orientation' in obj.keys():
                orientation: str = obj['orientation']

            # 根据物体位置和尺寸计算箭头的起点
            center_x = position[0]
            center_y = position[1]
            center_z = position[2]
            
            # 根据方向设置箭头的方向，假设 orientation 是上下左右的方向
            arrow_dx, arrow_dy, arrow_dz = 0, 0, 0

            if 'forward' in orientation.lower():
                arrow_length = size[1] / 2  # 箭头的长度
                arrow_dy = arrow_length  # 向上
            elif 'rearward' in orientation.lower():
                arrow_length = size[1] / 2  # 箭头的长度
                arrow_dy = -arrow_length  # 向下
            elif 'left' in orientation.lower():
                arrow_length = size[0] / 2  # 箭头的长度
                arrow_dx = -arrow_length  # 向左
            elif 'right' in orientation.lower():
                arrow_length = size[0] / 2  # 箭头的长度
                arrow_dx = arrow_length  # 向右
            else:
                logger.error(f"Invalid orientation: {orientation}")

            # 绘制箭头
            ax.quiver(center_x, center_y, center_z, arrow_dx, arrow_dy, arrow_dz, color='blue', length=arrow_length, normalize=True, arrow_length_ratio=0.2)

    
    # 设置轴的标签
    ax.set_xlabel('width')
    ax.set_ylabel('depth')
    ax.set_zlabel('height')

    # ax.set_xlim((-5, 5))
    # ax.set_ylim((-5, 5))
    # ax.set_zlim((-5, 5))
    ax.set_aspect(aspect='equal')
    ax.set_proj_type('ortho')

    # plt.show()
    return g_min_x, g_min_y, g_min_z, g_max_x, g_max_y, g_max_z

def visualize_objects(objects, idx):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    visualize_view(objects, ax1, elev=90, azim=-90)
    ax1.set_title("Top View")

    ax2 = fig.add_subplot(132, projection='3d')
    visualize_view(objects, ax2, elev=0, azim=0)
    ax2.set_title("Side View")
    
    ax3 = fig.add_subplot(133, projection='3d')
    visualize_view(objects, ax3, elev=0, azim=90)
    ax3.set_title("Front View")

    fig.suptitle(f"""#{idx}
anchor: color=cyan size={objects[0]['size']} location={objects[0]['location']} name={objects[0]['name']}
target: color=red size={objects[1]['size']} location={objects[1]['location']} name={objects[1]['name']}
relationship: {objects[1]['name']} -> {objects[1]['relationship']} -> {objects[0]['name']}"""
,fontsize=15) 
    plt.tight_layout()
    plt.show()

def visualize_objects_multi(objects, idx):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    visualize_view(objects, ax1, elev=90, azim=-90)
    ax1.set_title("Top View")

    ax2 = fig.add_subplot(132, projection='3d')
    visualize_view(objects, ax2, elev=0, azim=0)
    ax2.set_title("Side View")
    
    ax3 = fig.add_subplot(133, projection='3d')
    visualize_view(objects, ax3, elev=0, azim=90)
    ax3.set_title("Front View")

    anchor = None
    target = None
    for obj in objects:
        if obj['type'] == 'anchor':
            anchor = obj
        if obj['type'] == 'target':
            target = obj

    fig.suptitle(f"""#{idx}
anchor: color=cyan size={anchor['size']} location={anchor['location']} name={anchor['name']}
target: color=red size={target['size']} location={target['location']} name={target['name']}
relationship: {target['name']} -> {target['relationship']} -> {anchor['name']}"""
,fontsize=15) 
    plt.tight_layout()
    plt.show()


def render(objects, idx=None):
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(111, projection='3d')
    g_min_x, g_min_y, g_min_z, g_max_x, g_max_y, g_max_z = visualize_view(objects, ax1, elev=90, azim=-90)

    ax1.set_aspect('equal')
    ax1.autoscale(True, tight=True)

    ax1.set_xlim((g_min_x - 1, g_max_x + 1))
    ax1.set_xticks(np.arange(g_min_x - 1, g_max_x + 1 + 0.1, 0.1))
    # ax1.set_xlim((-1, 5))
    # ax1.set_xticks(np.arange(-1, 5 + 0.1, 0.1))
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45)

    ax1.set_ylim((g_min_y - 1, g_max_y + 1))
    ax1.set_yticks(np.arange(g_min_y - 1, g_max_y + 1 + 0.1, 0.1))
    # ax1.set_ylim((-1, 5))
    # ax1.set_yticks(np.arange(-1, 5 + 0.1, 0.1))
    ax1.set_ylabel('')

    ax1.set_zticks([])
    ax1.set_zlabel('')

    plt.tight_layout()
    ax1.set_position([0.0, 0.0, 1.0, 1.0])
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    buf = BytesIO()
    if idx is not None:
        plt.title(f'#{idx}')
    plt.savefig(buf, format='jpg')
    plt.show()
    plt.close(fig)
    
    # 通过PIL从缓冲区中加载图像并转换为NumPy数组
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)

    return img_np


def create_grid(objects, step, bound, wall=(0,0,0,0), visualize=False, level='cell', draw_emoji=False, direction_show=-1, avoid_anchor=True, required_coor=None, vis_size=1, title=None, avoid_all=False, adj_anchor=-1, render_size=0.48, draw_dir=False):
    objs = []
    emojis = Emojis()
    coverage = {'row':[], 'column':[], 'cell':[]}
    anchor_idx = None
    textual_layout = f"Region boundary: left={bound[0]}, right={bound[1]}, top={bound[2]}, bottom={bound[3]}\n"
    for idx, o in enumerate(objects):
        c_x, c_y = o['location'][0], o['location'][1]
        w, d = o['size'][0], o['size'][1]
        x, y = c_x - w / 2, c_y - d / 2
        orientation = o['orientation']

        if 'anchor' in o.keys() and o['anchor'] == True:
            anchor_idx = idx
            if 'color' not in o.keys():
                color = 'red'
            else:
                color = o['color']
        else:
            if 'color' not in o.keys():
                color = 'grey'
            else:
                color = o['color']
        objs.append(
            [x, y, w, d, o['name'], color, orientation]
        )
        coverage['row'].append([])
        coverage['column'].append([])
        coverage['cell'].append([])
        textual_orientation = 'forward' if o['orientation'] == 0 else 'rearward' if o['orientation'] == 1 else 'left' if o['orientation'] == 2 else 'right'
        textual_layout += f"{o['name']}: location={o['location']} size={o['size']} orientation={textual_orientation}\n"

    origin_x_lim = (bound[0], bound[1])
    origin_y_lim = (bound[2], bound[3])

    x_lim = (bound[0] - step, bound[1] + step)
    y_lim = (bound[2] - step, bound[3] + step)
    
    # Create the grid
    N_x = len(np.arange((origin_x_lim[0]), (origin_x_lim[1]), step))
    N_y = len(np.arange((origin_y_lim[0]), (origin_y_lim[1]), step))

    figsize = (max(int(N_x * render_size), 1), max(int(N_y * render_size), 1))
    logger.debug(f'figsize: {figsize}, N_x: {N_x}, N_y: {N_y}, render_size: {render_size}')
    scale = 5 / figsize[0]
    figsize = (max(int(figsize[0] * scale), 1), max(int(figsize[1] * scale), 1))
    fig, ax = plt.subplots(figsize=figsize)
    exceed_x = 0
    exceed_y = 0
    if np.arange((origin_x_lim[0]), (origin_x_lim[1]), step)[-1] + step > origin_x_lim[1]:
        exceed_x = 1
    if np.arange((origin_y_lim[0]), (origin_y_lim[1]), step)[-1] + step > origin_y_lim[1]:
        exceed_y = 1
    
    idx_to_coor = {
        'row': {},
        'column': {},
        'cell': {}
    }
    emojis_name = [k for k, v in emojis]
    emojis_name.remove('wall')
    emojis_name.remove('boundary') # except wall and boundary
    shuffle(emojis_name)
    emoji_used = {}
    for idx_y, y in enumerate(np.arange((origin_y_lim[0]), (origin_y_lim[1]), step)):
        for idx_x, x in enumerate(np.arange((origin_x_lim[0]), (origin_x_lim[1]), step)):
            covering_objs = check_inside([x, y], step, objs)
            idx_row = N_y - idx_y - 1 - exceed_y
            idx_col = idx_x
            global_idx = "{:03d}".format(idx_row * N_x + idx_col)
            row_idx = "{:02d}".format(idx_row)
            col_idx = "{:02d}".format(idx_col)
            for c in covering_objs:
                coverage['cell'][c].append(global_idx)
                coverage['row'][c].append(row_idx)
                coverage['column'][c].append(col_idx)

    for idx_y, y in enumerate(np.arange((origin_y_lim[0]), (origin_y_lim[1]), step)):
        for idx_x, x in enumerate(np.arange((origin_x_lim[0]), (origin_x_lim[1]), step)):
            x_center_offset = step / 2
            y_center_offset = step / 2
            draw = True
            if idx_x == len(np.arange((origin_x_lim[0]), (origin_x_lim[1]), step)) - 1:
                if x + step > origin_x_lim[1]:
                    x_center_offset = (origin_x_lim[1] - x) / 2
                    draw = False
                    # continue
            if idx_y == len(np.arange((origin_y_lim[0]), (origin_y_lim[1]), step)) - 1:
                if y + step > origin_y_lim[1]:
                    y_center_offset = (origin_y_lim[1] - y) / 2
                    draw = False
                    # continue
            ax.add_patch(patches.Rectangle((x, y), x_center_offset * 2, y_center_offset * 2, edgecolor='black', facecolor='white'))
            idx_row = N_y - idx_y - 1 - exceed_y
            idx_col = idx_x
            global_idx = "{:03d}".format(idx_row * N_x + idx_col)
            row_idx = "{:02d}".format(idx_row)
            col_idx = "{:02d}".format(idx_col)
            if required_coor and draw:
                if required_coor['axis'] == 'row':
                    for coor in required_coor['coor']:
                        draw = False
                        if abs(coor - (y + y_center_offset)) < 1e-5:
                            draw = True
                            break
                else:
                    for coor in required_coor['coor']:
                        draw = False
                        if not abs(coor - (x + x_center_offset)) < 1e-5:
                            draw = True
                            break
                    
            if direction_show == 0 and row_idx >= min(coverage['row'][anchor_idx]): # top
                draw = False
            elif direction_show == 2 and row_idx <= max(coverage['row'][anchor_idx]): # bottom
                draw = False
            elif direction_show == 1 and col_idx <= max(coverage['column'][anchor_idx]): # right
                draw = False
            elif direction_show == 3 and col_idx >= min(coverage['column'][anchor_idx]): # left
                draw = False

            for c in range(len(coverage['cell'])):
                if global_idx in coverage['cell'][c]:
                    draw = False
                    break
            if level == 'cell':
                emoji_idx = global_idx
                if adj_anchor > 0:
                    min_distance = min([
                        abs(int(row_idx) - int(r)) + abs(int(col_idx) - int(c)) \
                            for (r, c) in list(zip(coverage['row'][anchor_idx], coverage['column'][anchor_idx]))
                    ])
                    if min_distance > adj_anchor:
                        draw = False
            elif level == 'row':
                emoji_idx = row_idx
                if avoid_all:
                    for c in range(len(coverage['row'])):
                        if row_idx in coverage['row'][c]:
                            draw = False
                            break
                else:
                    if avoid_anchor and row_idx in coverage['row'][anchor_idx]:
                        draw = False
                if adj_anchor > 0:
                    min_distance = min([
                        abs(int(row_idx) - int(r)) \
                            for (r, c) in list(zip(coverage['row'][anchor_idx], coverage['column'][anchor_idx]))
                    ])
                    if min_distance > adj_anchor:
                        draw = False
            elif level == 'column':
                emoji_idx = col_idx
                if avoid_all:
                    for c in range(len(coverage['column'])):
                        if col_idx in coverage['column'][c]:
                            draw = False
                            break
                else:
                    if avoid_anchor and col_idx in coverage['column'][anchor_idx]:
                        draw = False
                if adj_anchor > 0:
                    min_distance = min([
                        abs(int(col_idx) - int(c)) \
                            for (r, c) in list(zip(coverage['row'][anchor_idx], coverage['column'][anchor_idx]))
                    ])
                    if min_distance > adj_anchor:
                        draw = False
            
            idx_to_coor['cell'][int(global_idx)] = (x + x_center_offset, y + y_center_offset)
            idx_to_coor['row'][int(row_idx)] = y + y_center_offset
            idx_to_coor['column'][int(col_idx)] = x + x_center_offset
            if draw:
                if draw_emoji:
                    if int(emoji_idx) >= len(emojis_name):
                        render_emoji_name = emojis_name[int(emoji_idx) % len(emojis_name)] 
                        logger.debug(f'emoji_idx: {emoji_idx} is out of range, use {int(emoji_idx)} % {len(emojis_name)} instead')
                    else:
                        render_emoji_name = emojis_name[int(emoji_idx)] 
                    ax.imshow(mpimg.imread(f'render_emoji/{render_emoji_name}.png'), extent=(x, x + x_center_offset * 2, y, y + y_center_offset * 2), zorder=10)
                else:
                    render_emoji_name = 'None'
                if level == 'row':
                    emoji_used[render_emoji_name] = y + y_center_offset
                elif level == 'column':
                    emoji_used[render_emoji_name] = x + x_center_offset
                elif level == 'cell':
                    emoji_used[render_emoji_name] = (x + x_center_offset, y + y_center_offset)

    wall_location = [
        ((origin_x_lim[0], origin_y_lim[1]), origin_x_lim[1] - origin_x_lim[0], step),
        ((origin_x_lim[1], origin_y_lim[0]), step, origin_y_lim[1] - origin_y_lim[0]),
        ((origin_x_lim[0], origin_y_lim[0] - step), origin_x_lim[1] - origin_x_lim[0], step),
        ((origin_x_lim[0] - step, origin_y_lim[0]), step, origin_y_lim[1] - origin_y_lim[0]),
    ]
    if draw_emoji:
        for wall_idx, value in enumerate(wall):
            (x_org, y_org), width, height = wall_location[wall_idx]
            wall_or_boundary = 'render_emoji/wall.png' if value == 1 else 'render_emoji/boundary.png'
            if width == step:
                for idx_y, y in enumerate(np.arange(origin_y_lim[0], origin_y_lim[1], step)):
                    ax.imshow(mpimg.imread(wall_or_boundary), extent=(x_org, x_org + step, y, y + step), zorder=10)
            else:
                for idx_x, x in enumerate(np.arange(origin_x_lim[0], origin_x_lim[1], step)):
                    ax.imshow(mpimg.imread(wall_or_boundary), extent=(x, x + step, y_org, y_org + step), zorder=10)

    for idx, c in enumerate(coverage['cell']):
        coverage['cell'][idx] = sorted(list(set(coverage['cell'][idx])))
        coverage['row'][idx] = sorted(list(set(coverage['row'][idx])))
        coverage['column'][idx] = sorted(list(set(coverage['column'][idx])))

    # Add objects to the grid based on input parameters
    for idx, o in enumerate(objs):
        x, y, width, height, label, color, orientation = o
        ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor='black', facecolor=color, alpha=0.1 if color == 'white' else 1.0))
        plt.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=20, color='black')
        
        # 添加箭头标注来指示方向
        if orientation is not None and draw_dir:
            # 计算箭头起点（物体中心）
            center_x = x + width/2
            center_y = y + height/2
            
            # 设置箭头长度为矩形较短边的1/3
            arrow_length = min(width, height) / 3
            
            # 根据orientation设置箭头方向
            dx, dy = 0, 0
            if orientation == 0:  # 向上
                dx, dy = 0, arrow_length
            elif orientation == 1:  # 向右
                dx, dy = arrow_length, 0
            elif orientation == 2:  # 向下
                dx, dy = 0, -arrow_length
            elif orientation == 3:  # 向左
                dx, dy = -arrow_length, 0
            
            # 绘制箭头
            ax.arrow(center_x, center_y, dx, dy, head_width=arrow_length/3, 
                    head_length=arrow_length/2, fc='blue', ec='blue', width=arrow_length/10)
        
    if title:
        ax.set_title(title)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().set_aspect('equal', adjustable='box')

    buf = BytesIO()
    plt.savefig(buf, format='jpg')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)
    if visualize:
        fig, ax = plt.subplots(figsize=[int(x * vis_size) for x in figsize])
        logger.debug(f'Image resolution: {img_np.shape}')
        ax.imshow(img_np, interpolation='nearest')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    return img_np, textual_layout, coverage, idx_to_coor, emoji_used


def check_inside(grid, step, objects):
    x_g, y_g = grid
    grid_box = [
        (x_g, y_g),
        (x_g + step, y_g),
        (x_g, y_g + step),
        (x_g + step, y_g + step),
    ]
    # def _check_axis(axis_obj_1, axis_obj_2):
    #     return max(min(axis_obj_1), min(axis_obj_2)) < min(max(axis_obj_1), max(axis_obj_2))
    def _check_axis(axis_obj_1, axis_obj_2):
        """检查两个box在某个轴上的投影是否有交集，返回交集的范围"""
        start = max(min(axis_obj_1), min(axis_obj_2))  # 交集的起始点
        end = min(max(axis_obj_1), max(axis_obj_2))    # 交集的终止点
        if start < end:
            return start, end  # 如果有交集，返回区间 (start, end)
        return None

    def _calculate_intersection_area(box_1, box_2):
        """计算两个box的相交面积"""
        # 获取box的x和y坐标
        x_1, y_1 = list(zip(*box_1))
        x_2, y_2 = list(zip(*box_2))
        
        # 计算x轴的交集
        x_intersection = _check_axis(x_1, x_2)
        # 计算y轴的交集
        y_intersection = _check_axis(y_1, y_2)
        
        # 如果x轴和y轴都有交集，计算相交面积
        if x_intersection and y_intersection:
            x_start, x_end = x_intersection
            y_start, y_end = y_intersection
            width = x_end - x_start
            height = y_end - y_start
            return width * height  # 面积 = 宽度 * 高度
        return 0 

    def _have_collapse(box_1, box_2):
        return _calculate_intersection_area(box_1, box_2) > 1e-4
        # return _check_axis(x_1, x_2) and _check_axis(y_1, y_2)

    res = []
    for idx, o in enumerate(objects):
        x, y, width, height, label, color, orientation = o
        obj_box = [
            (x, y),
            (x + width, y),
            (x, y + height),
            (x + width, y + height),
        ]
        if _calculate_intersection_area(grid_box, obj_box) > 1e-4:
            res.append(idx)
    return res