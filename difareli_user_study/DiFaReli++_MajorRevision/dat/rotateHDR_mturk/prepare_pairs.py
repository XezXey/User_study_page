import numpy as np
import glob, os, json, tqdm, sys
import argparse, subprocess
import pandas as pd
import numpy as np
import torch as th
import json, os, glob
import imageio
import numpy as np
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from PIL import Image
sys.path.append('/home/mint/Dev/DiFaReli/User_study_page/difareli_user_study/DiFaReli++_MajorRevision/')
from mint_logger import createLogger
logger = createLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--sample_json', type=str, required=True)
parser.add_argument('--candi_json', type=str, required=True)
parser.add_argument('--axis', type=int, required=True, choices=[1, 2])
parser.add_argument('--srcC', action='store_true', default=False, help='Whether to use srcC model for difareli++')
args = parser.parse_args()

def face_segment(segment_part, mask_path):
    face_segment_anno = imageio.v2.imread(mask_path)

    face_segment_anno = np.array(face_segment_anno)
    bg = (face_segment_anno == 0)
    skin = (face_segment_anno == 1)
    l_brow = (face_segment_anno == 2)
    r_brow = (face_segment_anno == 3)
    l_eye = (face_segment_anno == 4)
    r_eye = (face_segment_anno == 5)
    eye_g = (face_segment_anno == 6)
    l_ear = (face_segment_anno == 7)
    r_ear = (face_segment_anno == 8)
    ear_r = (face_segment_anno == 9)
    nose = (face_segment_anno == 10)
    mouth = (face_segment_anno == 11)
    u_lip = (face_segment_anno == 12)
    l_lip = (face_segment_anno == 13)
    neck = (face_segment_anno == 14)
    neck_l = (face_segment_anno == 15)
    cloth = (face_segment_anno == 16)
    hair = (face_segment_anno == 17)
    hat = (face_segment_anno == 18)
    face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip))

    if segment_part == 'faceseg_bg':
        seg_m = bg
    elif segment_part == 'faceseg_fg':
        seg_m = ~bg
    else: raise NotImplementedError(f"Segment part: {segment_part} is not found!")
    
    out = seg_m
    return out
            
def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        frame_idx = os.path.splitext(p.split('/')[-1].split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_anno.append(int(frame_idx))
    sorted_idx = np.argsort(frame_anno)
    sorted_path_list = []
    for idx in sorted_idx:
      sorted_path_list.append(path_list[idx])
    return sorted_path_list

def blending_mask(img_path, mask_path, hdr_from_bg_path):
    if isinstance(mask_path, np.ndarray):
        mask = mask_path / 255.0
    else:
        mask = imageio.v2.imread(mask_path) / 255.0 #[256, 256]
        
    if isinstance(img_path, np.ndarray):
        img = img_path / 255.0
    else:
        img = imageio.v2.imread(img_path).astype(np.float32) / 255.0
        
    blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    # Apply erosion
    kernel = np.ones((3,3),np.uint8)
    eroded_mask = cv2.erode(blurred_mask, kernel, iterations = 1)
    mask = eroded_mask [..., np.newaxis]
    
    bg = imageio.v2.imread(hdr_from_bg_path).astype(np.float32) / 255.0
    # alpha blending
    out = img * mask + bg * (1 - mask)
    out = (out * 255).astype(np.uint8)
    return out

method = [
    f"neural_gaffer_{'azimuth' if args.axis==1 else 'roll'}", 
    f"difareli++_axis{args.axis}_color_SD75_{'srcC' if args.srcC else '0.8C'}_Lmax10"
]
print("="*100)
print("[#] Methods to compare: ", method)
print("="*100)

meta = json.load(open(args.candi_json))
sample = json.load(open(args.sample_json))
out_dir = "./user_study_rotateHDR_pairs_prepared_MajorRevision/"
os.makedirs(out_dir, exist_ok=True)

if args.axis == 1:
    print("[#] Prepare for Axis 1 (Azimuth) rotation results...")
    hdr_map = ["125_hdrmaps_com_free_2K", "064_hdrmaps_com_free_2K", "117_hdrmaps_com_free_2K"]
else:
    print("[#] Prepare for Axis 2 (Roll) rotation results...")
    hdr_map = ["rotated_125_hdrmaps_com_free_2K", "rotated_064_hdrmaps_com_free_2K", "rotated_117_hdrmaps_com_free_2K"]

sota_key = f"neural_gaffer_{'azimuth' if args.axis==1 else 'roll'}"

n_pairs = 20
count = 1
# for pid, dat in sample['pair'].items():
for pid, dat in tqdm.tqdm(sample['pair'].items(), total=len(sample['pair'])):
    if count > n_pairs:
        break
    
    src = dat['src']
    dst = dat['dst']
    os.system(f"cp /data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/{src} {out_dir}/pair{count}/input_pair{count}.jpg")
    
    for hidx, hdr in enumerate(hdr_map):
        out_combined = []
        fail = False
        for m in method:
            if m == sota_key:
                path = meta[m]['res_dir']
                alias = meta[m]['alias']
                frame_path = f"{path}/{src.split('.')[0]}/{hdr}/"
                frames = sorted(glob.glob(f"{frame_path}/pred_*.png"))
                if len(frames) < 1:
                    print(f"[#] Skip {pid}-{src.split('.')[0]} for {m} since no result is found on {hdr}.")
                    fail = True
                    break
                target_hdr = sorted(glob.glob(f"{path}/{src.split('.')[0]}/{hdr}/target_*.png"))
                mask = f"/data/mint/DPM_Dataset/Dataset_For_Baseline/NeuralGaffer/input_subject_finale/preprocessed/mask/{src.split('.')[0]}.png"
            elif m == f"difareli++_axis{args.axis}_color_SD75_{'srcC' if args.srcC else '0.8C'}_Lmax10":
                path = meta[m]['res_dir']
                alias = meta[m]['alias']
                frame_path = f"{path}/{hdr}/src={src}/dst={dst}/Lerp_1000/n_frames=60/"
                frames = sort_by_frame(glob.glob(f"{frame_path}/res_frame*.png"))[1:][::-1]
                if len(frames) < 1:
                    print(f"Skip {pid}-{src.split('.')[0]} for {m} since no result is found on {hdr}.")
                    fail = True
                    break
                target_hdr = sorted(glob.glob(f"{meta[sota_key]['res_dir']}/{src.split('.')[0]}/{hdr}/target_*.png"))
                mask = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/face_segment_with_pupil/valid/anno/anno_{src.split('.')[0]}.png"
                mask = face_segment('faceseg_fg', mask) * 1.0
                
            # Reuse the target hdr as background for difareli++ too, but need to rescale first
            if "difareli++" in m:
                frames_tmp = [imageio.v2.imread(f) for f in frames] # 0 - 255
                mask_tmp = mask
                out_frames = []
                out_mask = []
                for i in range(len(frames_tmp)):
                    proc_img = frames_tmp[i]
                    proc_img = np.concatenate([proc_img, mask_tmp[..., np.newaxis] * 255], axis=-1)
                    
                    x, y, w, h = cv2.boundingRect((mask_tmp*255).astype(np.uint8))
                    max_size = max(w, h)
                    ratio = 0.75
                    side_len = int(max_size / ratio)
                    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
                    center = side_len//2
                    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = proc_img[y:y+h, x:x+w]
                    rgba = np.array(Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS))
                    # rgba is 0 - 255 for each channel
                    rgba_arr = np.array(rgba) / 255.0   # normalized to 0 - 1
                    rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])    # 256, 256, 3; 0-1
                    mask = rgba_arr[...,-1:]    # 256, 256, 1; 0-1
                    out_frames.append((rgb * 255).astype(np.uint8))
                    out_mask.append((mask[...,0] * 255).astype(np.uint8))
                frames = out_frames
                mask = out_mask[0]
                
            with mp.Pool(5) as p:
                out_frames = p.starmap(blending_mask, (zip(frames, [mask]*len(frames), target_hdr)))
            out_frames = np.stack(out_frames, axis=0)
            os.makedirs(f"{out_dir}/pair{count}/{alias}/{hdr}/", exist_ok=True)
            for i, f in enumerate(out_frames):
                Image.fromarray(f).save(f"{out_dir}/pair{count}/{alias}/{hdr}/res_frame{i:03d}.png")
    
        if not fail:
            # Create video for each method with target hdr as background
            vid_path = f"{out_dir}/pair{count}/"
            for m in method:
                alias = meta[m]['alias']
                frame_path = f"{out_dir}/pair{count}/{alias}/{hdr}/"
                frames = sort_by_frame(glob.glob(f"{frame_path}/res_frame*.png"))
                name = f"{alias.split('_')[0]}_pair{count}_{hdr}_axis{args.axis}.mp4"
                video_name = f"{vid_path}/{name}"
                cmd = f"ffmpeg -y -framerate 24 -i {frame_path}/res_frame%03d.png -c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video_name}"
                # subprocess.call(cmd, shell=True)
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            # Copy target hdr to the pair folder
            target_hdr = sorted(glob.glob(f"{meta[sota_key]['res_dir']}/{src.split('.')[0]}/{hdr}/target_*.png"))
            os.makedirs(f"{vid_path}/target_a{args.axis}/{hdr}/", exist_ok=True)
            for t in target_hdr:
                Image.open(t).save(f"{vid_path}/target_a{args.axis}/{hdr}/{os.path.basename(t)}")
            # Write video of target hdr
            cmd = f"ffmpeg -y -framerate 24 -i {vid_path}/target_a{args.axis}/{hdr}/target_%04d.png -c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {vid_path}/hdr_pair{count}_{hdr}_axis{args.axis}.mp4"
            # Hide the output of ffmpeg
            # subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # subprocess.call(cmd, shell=True)

    count += 1
        
            # out_combined = np.concatenate(out_combined, axis=0)
            # # Resize cond's widgth to match out_combined and append to the bottom
            # cond = cv2.resize(cond, (out_combined.shape[1], int(cond.shape[0] * out_combined.shape[1] / cond.shape[1])), interpolation=cv2.INTER_LANCZOS4)
            # out_combined = np.concatenate([out_combined, cond], axis=0)
            # os.makedirs(f"{out_dir}/{hdr}/", exist_ok=True)
            