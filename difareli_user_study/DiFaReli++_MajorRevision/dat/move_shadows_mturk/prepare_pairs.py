import numpy as np
import glob, os, json, tqdm, sys
import pytorch_lightning as pl
import argparse, subprocess
import gen_ball
import pandas as pd
from collections import defaultdict
from PIL import Image
sys.path.append('/home/mint/Dev/DiFaReli/User_study_page/difareli_user_study/DiFaReli++_MajorRevision/')
from mint_logger import createLogger
logger = createLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--gen_pairs', action='store_true', default=False)
parser.add_argument('--sample_json', type=str, required=True)
parser.add_argument('--candi_json', type=str, default='./candidates.json')
parser.add_argument('--add_wm', action='store_true', default=False)
args = parser.parse_args()
    
def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        # frame_idx = os.path.splitext(p.split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_idx = os.path.splitext(p.split('/')[-1].split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_anno.append(int(frame_idx))
    sorted_idx = np.argsort(frame_anno)
    sorted_path_list = []
    for idx in sorted_idx:
      sorted_path_list.append(path_list[idx])
    return sorted_path_list

def read_light_params():
    def read_params(path):
        params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
        params.rename(columns={0:'img_name'}, inplace=True)
        params = params.set_index('img_name').T.to_dict('list')
        return params

    def swap_key(params):
        params_s = defaultdict(dict)
        for params_name, v in params.items():
            for img_name, params_value in v.items():
                params_s[img_name][params_name] = np.array(params_value).astype(np.float64)

        return params_s
    
    params = read_params('/data/mint/DPM_Dataset/ffhq_256_with_anno/params/valid/ffhq-valid-light-anno.txt')
    params = swap_key({'light':params})
    return params

def face_segment(segment_part, face_segment_anno):

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
    face_noears = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, nose, mouth, u_lip, l_lip))

    if segment_part == 'faceseg_face':
        seg_m = face
    elif segment_part == 'faceseg_face&noears':
        seg_m = face_noears
    elif segment_part == 'faceseg_head':
        seg_m = (face | neck | hair)
    elif segment_part == 'faceseg_nohead':
        seg_m = ~(face | neck | hair)
    elif segment_part == 'faceseg_face&hair':
        seg_m = ~bg
    elif segment_part == 'faceseg_bg_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth) 
    elif segment_part == 'faceseg_bg&ears_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth) | (l_ear | r_ear | ear_r)
    elif segment_part == 'faceseg_bg':
        seg_m = bg
    elif segment_part == 'faceseg_bg&noface':
        seg_m = (bg | hair | hat | neck | neck_l | cloth)
    elif segment_part == 'faceseg_hair':
        seg_m = hair
    elif segment_part == 'faceseg_faceskin':
        seg_m = skin
    elif segment_part == 'faceseg_faceskin&nose':
        seg_m = (skin | nose)
    elif segment_part == 'faceseg_face_noglasses':
        seg_m = (~eye_g & face)
    elif segment_part == 'faceseg_face_noglasses_noeyes':
        seg_m = (~(l_eye | r_eye) & ~eye_g & face)
    elif segment_part == 'faceseg_eyes&glasses':
        seg_m = (l_eye | r_eye | eye_g)
    elif segment_part == 'faceseg_eyes':
        seg_m = (l_eye | r_eye)
    else: raise NotImplementedError(f"Segment part: {segment_part} is not found!")
    
    out = seg_m
    return out

def wm_area(img, mask, wm, alpha):
    # Create red image and blend with alpha factor on "img" on specific masking area
    blended = (wm * alpha + (img * (1 - alpha)))
    out = (blended * ~mask[..., None]) + (img * mask[..., None])
    out = np.clip(out.astype(np.uint8), 0, 255)
    return out
    

def gen_pairs():
    #NOTE: This is for original images
    dataset_path = '/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/'
    params = read_light_params()
    
    with open(args.candi_json, 'r') as f:
        candi = json.load(f)
    with open(args.sample_json, 'r') as f:
        sample_pairs = json.load(f)['pair']

    out_f = './user_study_rotateSH_pairs_prepared'
    os.makedirs(f'{out_f}', exist_ok=True)
    i = 0
    # for pair in tqdm.tqdm(eval_pairs):
    for pair_id in tqdm.tqdm(sample_pairs.values()):
        pair = ['src=' + pair_id['src'], 'dst=' + pair_id['dst']]
        os.makedirs(f'./{out_f}/pair{i+1}', exist_ok=True)
        # Load mask img
        for m in candi.keys():
            n_frames = candi[m]['n_frames']
            m_path = candi[m]['full_path']
            do_n1 = candi[m]['do_n1']
            res = candi[m].get('res', '')
            axis = 1 if 'a1' in m else 2
            
            # Frames
            # if len(n_frames_dir) > 0:
            #     n_frames_dir = n_frames_dir[0].split('/')[-1]
            #     frames = glob.glob(f"{m_path}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/{n_frames_dir}/res_frame*.png")
            # else: 
            #     frames = glob.glob(f"{m_path}/{pair[0]}/{pair[1]}/res_frame*.png")
            
            # # print(frames)
            # frames = sort_by_frame(frames)
            
            # relit_dat_path = frames[-1]
            # Relighting vids
            if do_n1:
                if not os.path.exists(f"{m_path}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/n_frames={n_frames}/{res}/"):
                    logger.warning(f"Path not found: {m_path}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/n_frames={n_frames}/{res}/")
                    continue
                    
                frames = glob.glob(f"{m_path}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/n_frames={n_frames}/{res}/res_frame*.png")
                # print(f"{m_path}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/n_frames={n_frames}/{res}/res_frame*.png")
                frames = sort_by_frame(frames)[1:] # Skip the first frame
                # FFMPEG from frames (list of image paths) to video, high quality, 24fps and no lossy compression
                os.makedirs(f'./{out_f}/pair{i+1}/{m}_frames/', exist_ok=True)
                for frame in frames:
                    idx = frame.split('/')[-1].split('frame')[-1].split('.')[0]
                    if args.add_wm:
                        # Facial part
                        mask_anno = np.array(Image.open(f"/data/mint/DPM_Dataset/ffhq_256_with_anno/face_segment/valid/anno/anno_{pair[0].split('=')[-1].replace('.jpg', '.png')}"))
                        mask_facial = face_segment('faceseg_face&noears', mask_anno)
                        mask_bg = face_segment('faceseg_bg', mask_anno)
                        
                        #NOTE: Watermark
                        wm = Image.open("./bw_overlay.jpg").convert('RGB')
                        wm = np.array(wm.resize((256, 256))).astype(np.uint8)

                        img = np.array(Image.open(frame))
                        alpha = 0.65
                        blend_img = wm_area(img, mask_facial, wm, alpha)
                        Image.fromarray(blend_img).save(f"./{out_f}/pair{i+1}/{m}_frames/res_frame{int(idx):04d}_face.png")
                        
                        blend_img = wm_area(img, ~mask_bg, wm, alpha)
                        Image.fromarray(blend_img).save(f"./{out_f}/pair{i+1}/{m}_frames/res_frame{int(idx):04d}.png")
                    else:
                        os.system(f"cp {frame} ./{out_f}/pair{i+1}/{m}_frames/res_frame{int(idx):04d}.png")
                fw_path = f"./{out_f}/pair{i+1}/{m}_frames/{m}_pair{i+1}_axis1_fw.mp4"
                bw_path = f"./{out_f}/pair{i+1}/{m}_frames/{m}_pair{i+1}_axis1_bw.mp4"
                final_path = f"./{out_f}/pair{i+1}/{m.replace(f'_a{axis}', '')}_pair{i+1}_axis{axis}.mp4"

                # Forward video
                fw_cmd = (
                    f"ffmpeg -r 24 -i ./{out_f}/pair{i+1}/{m}_frames/res_frame%04d.png "
                    f"-c:v libx264 -crf 17 -preset slow -vf 'fps=24,format=yuv420p' -y {fw_path}"
                )

                # Backward video
                bw_cmd = (
                    f"ffmpeg -r 24 -i ./{out_f}/pair{i+1}/{m}_frames/res_frame%04d.png "
                    f"-vf 'reverse,fps=24,format=yuv420p' -c:v libx264 -crf 17 -preset slow -y {bw_path}"
                )

                # Concatenate videos
                concat_cmd = (
                    f"ffmpeg -i {fw_path} -i {bw_path} "
                    f"-filter_complex '[0:v] [1:v] concat=n=2:v=1 [v]' -map '[v]' -c:v libx264 -crf 17 -preset slow -y {final_path}"
                )
                # Run commands
                for cmd in [fw_cmd, bw_cmd, concat_cmd]:
                    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode != 0:  # Check for errors
                        print(f"Error executing command {cmd}:\n{result.stderr}")
            else:
                res_vid = f"{m_path}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/n_frames={n_frames}/{res}/res_rt_nof.mp4"
                os.system(f'cp {res_vid} ./{out_f}/pair{i+1}/{m}_pair{i+1}_axis1.mp4')
            
            #NOTE: Input
            # os.system(f'cp {input_dat_path} ./{out_f}/pair{i+1}/input_pair{i+1}.png')
            os.system(f"cp {dataset_path}/{pair[0].split('=')[-1]} ./{out_f}/pair{i+1}/input_pair{i+1}.jpg")
                
            #NOTE: Target
            os.system(f"cp {dataset_path}/{pair[1].split('=')[-1]} ./{out_f}/pair{i+1}/target_pair{i+1}.jpg")
            
            #NOTE: SH
            # gen_ball.drawSH(params[pair[1].split('=')[-1]]['light'], f"./{out_f}/pair{i+1}/sh_pair{i+1}.jpg")
            # print(f"/data/mint/DPM_Dataset/Dataset_For_Baseline/ffhq_user_study/axis={axis}/valid/*_{pair[0]}_{pair[1]}")
            ball_path = glob.glob(f"/data/mint/DPM_Dataset/Dataset_For_Baseline/ffhq_user_study/axis={axis}/valid/*_{pair[0]}_{pair[1]}")[0]
            ball_path = f"{ball_path}/n_step={n_frames}/ball/"
            if do_n1:
                ball_frames = sorted(glob.glob(f"{ball_path}/m_*.png"))[1:]
                os.makedirs(f'./{out_f}/pair{i+1}/ball/', exist_ok=True)
                for frame in ball_frames:
                    os.system(f"cp {frame} ./{out_f}/pair{i+1}/ball/{frame.split('/')[-1]}")
                fw_cmd = (
                    f"ffmpeg -r 24 -i ./{out_f}/pair{i+1}/ball/m_%03d.png "
                    f"-c:v libx264 -crf 17 -preset slow -vf 'fps=24,format=yuv420p' -y ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_fw.mp4"
                )
                bw_cmd = (
                    f"ffmpeg -r 24 -i ./{out_f}/pair{i+1}/ball/m_%03d.png "
                    f"-vf 'reverse,fps=24,format=yuv420p' -c:v libx264 -crf 17 -preset slow -y ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_bw.mp4"
                )
                concat_cmd = (
                    f"ffmpeg -i ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_fw.mp4 -i ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_bw.mp4 "
                    f"-filter_complex '[0:v] [1:v] concat=n=2:v=1 [v]' -map '[v]' -c:v libx264 -crf 17 -preset slow -y ./{out_f}/pair{i+1}/sh_pair{i+1}_axis{axis}.mp4"
                )
                for cmd in [fw_cmd, bw_cmd, concat_cmd]:
                    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode != 0:
                        print(f"Error executing command {cmd}:\n{result.stderr}")

            # os.makedirs(f'./{out_f}/pair{i+1}/ball/', exist_ok=True)
            # os.system(f"cp {ball_path} ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_fw.mp4")
            # # FFMPEG to Reverse the ball video and concat
            # ball_bw_cmd = f"ffmpeg -i ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_fw.mp4 -vf reverse ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_bw.mp4"
            # ball_concat_cmd = f"ffmpeg -i ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_fw.mp4 -i ./{out_f}/pair{i+1}/ball/sh_pair{i+1}_axis{axis}_bw.mp4 -filter_complex '[0:v] [1:v] concat=n=2:v=1 [v]' -map '[v]' -c:v libx264 -crf 17 -preset slow -y ./{out_f}/pair{i+1}//sh_pair{i+1}_axis{axis}.mp4"
            # for cmd in [ball_bw_cmd, ball_concat_cmd]:
            #     result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            #     if result.returncode != 0:
            #         print(f"Error executing command {cmd}:\n{result.stderr}")

            
        # Dump empty text file that named as the pair name
        with open(f"./{out_f}/pair{i+1}/{pair[0]}_{pair[1]}.txt", 'w') as f:
            f.write('')
        i += 1
            
if __name__ == '__main__':
    pl.seed_everything(args.seed)
    gen_pairs()
