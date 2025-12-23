import argparse
import os
import sys
from datetime import datetime
import mediapipe as mp
import numpy as np
import cv2
import torch
from skimage.transform import resize
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL, AutoencoderTiny
from src.scheduler.scheduler_ddim import DDIMScheduler
import random
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline, Pose2VideoPipeline_Stream
from src.utils.util import save_videos_grid, crop_face
from decord import VideoReader
from diffusers.utils.import_utils import is_xformers_available

from src.models.motion_encoder.encoder import MotEncoder
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/prompts/personalive_offline.yaml')
    parser.add_argument("--name", type=str, default='personalive_offline')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_xformers", type=bool, default=True)
    parser.add_argument("--stream_gen", type=bool, default=True, help='use streaming generation strategy to reduce VRAM usage.')
    parser.add_argument("--reference_image", type=str, default='', help='Path to reference image. If provided, overrides test_cases from config.')
    parser.add_argument("--driving_video", type=str, default='', help='Path to driving video. If provided, overrides test_cases from config.')
    args = parser.parse_args()

    return args

def main(args):
    device = args.device
    print('device', device)
    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(config.vae_path).to(device, dtype=weight_dtype)
    # if use tiny VAE
    # vae_tiny = AutoencoderTiny.from_pretrained(config.vae_tiny_path).to(device, dtype=weight_dtype)

    infer_config = OmegaConf.load(config.inference_config)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
    pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)
    pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()
    
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(
        OmegaConf.load(config.inference_config).noise_scheduler_kwargs
    )
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)
    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
    )
    reference_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'reference_unet'),
            map_location="cpu",
        ),
        strict=True,
    )
    motion_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_encoder'),
            map_location="cpu",
        ),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'pose_guider'),
            map_location="cpu",
        ),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'temporal_module'),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_extractor'),
            map_location="cpu",
        ),
        strict=False,
    )
    
    if args.use_xformers:
        if is_xformers_available(): 
            try:
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("Failed to enable xformers:", e)
        else:
            print("xformers is not available. Make sure it is installed correctly.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    if args.stream_gen:
        pipeline = Pose2VideoPipeline_Stream
    else:
        pipeline = Pose2VideoPipeline
    
    pipe = pipeline(
        vae=vae,
        # vae_tiny=vae_tiny,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        pose_encoder=pose_encoder,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    date_str = datetime.now().strftime("%Y%m%d")
    if args.name is None:
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{date_str}--{time_str}"
    else:
        save_dir_name = f"{date_str}--{args.name}"
    save_vid_dir = os.path.join('results', save_dir_name, 'concat_vid')
    os.makedirs(save_vid_dir, exist_ok=True)
    save_split_vid_dir = os.path.join('results', save_dir_name, 'split_vid')
    os.makedirs(save_split_vid_dir, exist_ok=True)

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    if args.reference_image and args.driving_video:
        args.test_cases = {args.reference_image: [args.driving_video]}
    else:
        args.test_cases = OmegaConf.load(args.config)["test_cases"]

    for ref_image_path in list(args.test_cases.keys()):
        for pose_video_path in args.test_cases[ref_image_path]:
            video_name = os.path.basename(pose_video_path).split(".")[0]
            source_name = os.path.basename(ref_image_path).split(".")[0]

            vid_name = f"{source_name}_{video_name}.mp4"
            save_vid_path = os.path.join(save_vid_dir, vid_name)
            print(save_vid_path)
            if os.path.exists(save_vid_path):
                continue

            if ref_image_path.endswith('.mp4'):
                src_vid = VideoReader(ref_image_path)
                ref_img = src_vid[0].asnumpy()
                ref_img = Image.fromarray(ref_img).convert("RGB")
            else:
                ref_img = Image.open(ref_image_path).convert("RGB")

            control = VideoReader(pose_video_path)
            video_length = min(len(control) // 4 * 4, args.L)
            sel_idx = range(len(control))[:video_length]
            control = control.get_batch([sel_idx]).asnumpy() # N, H, W, C

            ref_image_pil = ref_img.copy()
            ref_patch = crop_face(ref_image_pil, face_mesh)
            ref_face_pil = Image.fromarray(ref_patch).convert("RGB")

            size = args.H
            generator = torch.Generator(device=device)
            generator.manual_seed(42)

            dri_faces = []
            ori_pose_images = []
            for idx_control, pose_image_pil in tqdm(enumerate(control[:video_length]), total=video_length, desc='cropping faces'):
                pose_image_pil = Image.fromarray(pose_image_pil).convert("RGB")
                ori_pose_images.append(pose_image_pil)
                dri_face = crop_face(pose_image_pil, face_mesh)
                dri_face_pil = Image.fromarray(dri_face).convert("RGB")
                dri_faces.append(dri_face_pil)

            face_tensor_list = []
            ori_pose_tensor_list = []
            ref_tensor_list = []

            for idx, pose_image_pil in enumerate(ori_pose_images):
                face_tensor_list.append(pose_transform(dri_faces[idx]))
                ori_pose_tensor_list.append(pose_transform(pose_image_pil))
                ref_tensor_list.append(pose_transform(ref_image_pil))

            ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
            ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)  # (c, f, h, w)

            face_tensor = torch.stack(face_tensor_list, dim=0)  # (f, c, h, w)
            face_tensor = face_tensor.transpose(0, 1).unsqueeze(0)

            ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)  # (f, c, h, w)
            ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

            gen_video = pipe(
                ori_pose_images,
                ref_image_pil,
                dri_faces,
                ref_face_pil,
                width,
                height,
                len(dri_faces),
                num_inference_steps=4,
                guidance_scale=1.0,
                generator=generator,
                temporal_window_size = 4,
                temporal_adaptive_step = 4,
            ).videos

            #Concat it with pose tensor
            video = torch.cat([ref_tensor, face_tensor, ori_pose_tensor, gen_video], dim=0)

            save_videos_grid(
                video,
                save_vid_path,
                n_rows=4,
                fps=25,
            )

            if True:
                save_vid_path = save_vid_path.replace(save_vid_dir, save_split_vid_dir)
                save_videos_grid(gen_video, save_vid_path, n_rows=1, fps=25, crf=18, audio_source=pose_video_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
