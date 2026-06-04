import sys
import argparse
import os
import cv2
import torch
import torchvision
import time
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from .controlnet.diffusers_cnet_txt2img import txt2imgControlNet
from .controlnet.diffusers_cnet_inpaint import inpaintControlNet
from .paint3d import utils
from .paint3d.models.textured_mesh import TexturedMeshModel
from .paint3d.dataset import init_dataloaders
from .paint3d.trainer import dr_eval, forward_texturing

device = torch.device("cuda")

sd_cfg = OmegaConf.load('Paint3D/controlnet/config/depth_based_inpaint_template.yaml')
depth_cnet = txt2imgControlNet(sd_cfg.txt2img)
inpaint_cnet = inpaintControlNet(sd_cfg.inpaint)


def inpaint_viewpoint(sd_cfg, cnet, save_result_dir, mesh_model, dataloaders, inpaint_view_ids=[(5, 6)]):
    # projection
    view_angle_info = {i:data for i, data in enumerate(dataloaders['train'])}
    inpaint_used_key = ["image", "depth", "uncolored_mask"]
    for i, one_batch_id in enumerate(inpaint_view_ids):
        one_batch_img = []
        for view_id in one_batch_id:
            data = view_angle_info[view_id]
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
            view_img_info = [outputs[k] for k in inpaint_used_key]
            one_batch_img.append(view_img_info)

        for i, img in enumerate(zip(*one_batch_img)):
            img = torch.cat(img, dim=3)
            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)
            t = '_'.join(map(str, one_batch_id))
            name = inpaint_used_key[i]
            if name == "uncolored_mask":
                img[img>0] = 1
            save_path = os.path.join(save_result_dir, f"view_{t}_{name}.png")
            utils.save_tensor_image(img, save_path=save_path)

    # inpaint view point
    txt_cfg = sd_cfg.txt2img
    img_cfg = sd_cfg.inpaint
    copy_list = ["prompt", "negative_prompt", "seed", ]
    for k in copy_list:
        img_cfg[k] = txt_cfg[k]

    for i, one_batch_id in enumerate(inpaint_view_ids):
        t = '_'.join(map(str, one_batch_id))
        rgb_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[0]}.png")
        depth_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[1]}.png")
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}.png")

        # pre-processing inpaint mask: dilate
        mask = cv2.imread(mask_path)
        dilate_kernel = 10
        mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}_d{dilate_kernel}.png")
        cv2.imwrite(mask_path, mask)

        img_cfg.image_path = rgb_path
        img_cfg.mask_path =  mask_path
        img_cfg.controlnet_units[0].condition_image_path = depth_path
        images = cnet.infernece(config=img_cfg)
        for i, img in enumerate(images):
            save_path = os.path.join(save_result_dir, f"view_{t}_rgb_inpaint_{i}.png")
            img.save(save_path)
    return images


def gen_init_view(sd_cfg, cnet, mesh_model, dataloaders, outdir, view_ids=[]):
    init_depth_map = []
    view_angle_info = {i: data for i, data in enumerate(dataloaders['train'])}
    for view_id in view_ids:
        data = view_angle_info[view_id]
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
        depth_render = outputs['depth']
        init_depth_map.append(depth_render)
        # Save individual depth map in RGB format
        depth_rgb = depth_render.repeat(1, 3, 1, 1)
        save_path = os.path.join(outdir, f"init_depth_render_{view_id}.png")
        utils.save_tensor_image(depth_rgb, save_path=save_path)

    # Create grid for visualization
    init_depth_map = torch.cat(init_depth_map, dim=0).repeat(1, 3, 1, 1)
    init_depth_map = torchvision.utils.make_grid(init_depth_map, nrow=2, padding=0)
    save_path = os.path.join(outdir, f"init_depth_render.png")
    utils.save_tensor_image(init_depth_map.unsqueeze(0), save_path=save_path)

    # post-processing depth for each view and the combined grid
    for view_id in view_ids:
        # Process individual depth maps
        depth_path = os.path.join(outdir, f"init_depth_render_{view_id}.png")
        depth_dilated = utils.dilate_depth_outline(depth_path, iters=5, dilate_kernel=3)
        save_path = os.path.join(outdir, f"init_depth_dilated_{view_id}.png")
        cv2.imwrite(save_path, depth_dilated)

    # Process the combined grid image
    depth_dilated = utils.dilate_depth_outline(os.path.join(outdir, f"init_depth_render.png"), iters=5, dilate_kernel=3)
    save_path = os.path.join(outdir, f"init_depth_dilated.png")
    cv2.imwrite(save_path, depth_dilated)

    p_cfg = sd_cfg.txt2img
    
    images = []
    # for view_id in view_ids:
    if True:
        p_cfg.controlnet_units[0].condition_image_path = os.path.join(outdir, f"init_depth_dilated.png")
        # if view_id == 0:
        #     p_cfg.prompt = 'front view, ' + p_cfg.prompt
        #     p_cfg.negative_prompt = 'back view, side view, ' + p_cfg.negative_prompt
        #     p_cfg.width = 512
        #     p_cfg.height = 512
        # else:
        #     p_cfg.prompt = 'back view, ' + p_cfg.prompt
        #     p_cfg.negative_prompt = 'front view, side view' + p_cfg.negative_prompt
        #     p_cfg.width = 512
        #     p_cfg.height = 512

        for i, img in enumerate(cnet.infernece(config=p_cfg)):
            images.append(img)
            save_path = os.path.join(outdir, f'init-img-{i}-{view_id}.png')
            img.save(save_path)

    # 横向拼接所有生成的图像
    if images:
        # 获取第一张图像的尺寸
        img_width, img_height = images[0].size
        
        # 创建拼接后的图像
        total_width = img_width * len(images)
        concatenated_img = Image.new('RGB', (total_width, img_height))
        
        # 逐个粘贴图像
        for i, img in enumerate(images):
            concatenated_img.paste(img, (i * img_width, 0))
        
        # 保存拼接后的图像
        concat_save_path = os.path.join(outdir, 'init-img-0.png')
        concatenated_img.save(concat_save_path)
    return [concatenated_img]


def init_process(opt):
    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)

    pathdir, filename = Path(opt.render_config).parent, Path(opt.render_config).stem
    sys.path.append(str(pathdir))
    render_cfg = __import__(filename, ).TrainConfig()
    utils.seed_everything(render_cfg.optim.seed)

    sd_cfg = OmegaConf.load(opt.sd_config)
    render_cfg.log.exp_path = str(outdir)
    if opt.prompt is not None:
        sd_cfg.txt2img.prompt = opt.prompt
    if opt.ip_adapter_image_path is not None:
        sd_cfg.txt2img.ip_adapter_image_path = opt.ip_adapter_image_path
        sd_cfg.inpaint.ip_adapter_image_path = opt.ip_adapter_image_path
    if opt.mesh_path is not None:
        render_cfg.guide.shape_path = opt.mesh_path
    if opt.texture_path is not None:
        render_cfg.guide.initial_texture = opt.texture_path
        img = Image.open(opt.texture_path)
        render_cfg.guide.texture_resolution = img.size
    return sd_cfg, render_cfg


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd_config",
        type=str,
        default="stable-diffusion/v2-inpainting-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--render_config",
        type=str,
        default=" ",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="prompt",
        default=None,
    )
    parser.add_argument(
        "--ip_adapter_image_path",
        type=str,
        help="prompt",
        default=None,
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        help="path of mesh",
        default=None,
    )
    parser.add_argument(
        "--texture_path",
        type=str,
        help="path of texture image",
        default=None,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/inpainting-samples"
    )

    opt = parser.parse_args()
    return opt


@torch.inference_mode()
def retexture_stage1(output_dir, prompt, obj_path):
    opt = edict(
        sd_config='Paint3D/controlnet/config/depth_based_inpaint_template.yaml',
        render_config='Paint3D/paint3d/config/train_config_paint3d.py',
        mesh_path=obj_path,
        prompt=prompt,
        outdir=output_dir,
        ip_adapter_image_path=None,
        texture_path=None,
    )
    sd_cfg, render_cfg = init_process(opt)

    dataloaders = init_dataloaders(render_cfg, device)
    mesh_model = TexturedMeshModel(cfg=render_cfg, device=device)

    # ===  2. init view generation

    init_images = gen_init_view(
        sd_cfg=sd_cfg,
        cnet=depth_cnet,
        mesh_model=mesh_model,
        dataloaders=dataloaders,
        outdir=output_dir,
        view_ids=render_cfg.render.views_init,
    )

    for i, init_image in enumerate(init_images):
        output_dir = Path(output_dir) / f"res-{i}"
        output_dir.mkdir(exist_ok=True)
        #  back-projection init view
        mesh_model.initial_texture_path = None
        mesh_model.refresh_texture()
        view_imgs = utils.split_grid_image(img=np.array(init_image), size=(1, 2))
        forward_texturing(
            cfg=render_cfg,
            dataloaders=dataloaders,
            mesh_model=mesh_model,
            save_result_dir=output_dir,
            device=device,
            view_imgs=view_imgs,
            view_ids=render_cfg.render.views_init,
            verbose=False,
        )
        # === 3. depth based inpaint
        for view_group in render_cfg.render.views_inpaint:   # cloth 4 view
            start_t = time.time()
            output_dir = Path(output_dir) / f"res-{i}"
            output_dir.mkdir(exist_ok=True)
            inpainted_images = inpaint_viewpoint(
                sd_cfg=sd_cfg,
                cnet=inpaint_cnet,
                save_result_dir=output_dir,
                mesh_model=mesh_model,
                dataloaders=dataloaders,
                inpaint_view_ids=[view_group],
            )


            start_t = time.time()
            view_imgs = []
            for img_t in inpainted_images:
                view_imgs.extend(utils.split_grid_image(img=np.array(img_t), size=(1, 2)))
            # print(f"forward_texturing {i} start, {time.time()}")
            forward_texturing(
                cfg=render_cfg,
                dataloaders=dataloaders,
                mesh_model=mesh_model,
                save_result_dir=output_dir,
                device=device,
                view_imgs=view_imgs,
                view_ids=view_group,
                verbose=False,
            )

        mesh_model.initial_texture_path = f"{output_dir}/albedo.png"
        mesh_model.refresh_texture()
        # dr_eval(
        #     cfg=render_cfg,
        #     dataloaders=dataloaders,
        #     mesh_model=mesh_model,
        #     save_result_dir=output_dir,
        #     valset=True,
        #     verbose=False,
        # )
        mesh_model.empty_texture_cache()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    retexture_stage1()
