import numpy as np
import glob, os, json, tqdm
import pytorch_lightning as pl
import argparse
import gen_ball
import pandas as pd
from collections import defaultdict
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--curr_ip', type=str, required=True)
parser.add_argument('--num_pairs', type=int, required=True)
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--do_mount', action='store_true', default=False)
parser.add_argument('--gen_pairs', action='store_true', default=False)
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

def mount_dat():
    with open('./candidates.json', 'r') as f:
        candi = json.load(f)
    
    if not os.path.exists('./mount'):
        os.makedirs('./mount')
        
    for name, info in candi.items():
        os.makedirs(f'./mount/{name}', exist_ok=True)
        to_path = f"./mount/{name}/"
        
        if info['ip'] == args.curr_ip:
            from_path = f"/{info['path']}/{info['dir']}/{info['pre_misc']}/*"
            print(f"[#] Symlink {from_path} to {to_path}")
            os.system(f'ln -s {from_path} {to_path}')
        else:
            from_path = f"{info['ip']}:/{info['path']}/{info['dir']}/{info['pre_misc']}"
            print(f"[#] Mount {from_path} to {to_path}")
            # if to_path is not empty, then umount it first
            if os.listdir(to_path):
                print(f"===> [#] {to_path} is not empty, umount it first")
                os.system(f'sudo umount {to_path}')
                
            print(f"===> [#] {to_path} is empty, mount it")
            os.system(f'sshfs -o ro {from_path} {to_path}')

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
    
    with open('./candidates.json', 'r') as f:
        candi = json.load(f)
    
    # Get pool based on Total Relighting
    # pool = [p.split('/')[-2:] for p in glob.glob('./mount/difareli/*/*')]
    pool = [p.split('/')[-2:] for p in glob.glob('./mount/total_relighting/*/*')]
    eval_pairs = []
    for i in tqdm.tqdm(range(args.num_pairs)):
        all_exist = False
        while not all_exist:
            pair = pool[np.random.choice(np.arange(len(pool)), 1)[0]]
            all_exist = all([os.path.exists(f'./mount/{m}/{pair[0]}/{pair[1]}') for m in candi.keys()])
        pool.remove(pair)
        eval_pairs.append(pair)
    os.makedirs('./eval_pairs', exist_ok=True)
    i = 0
    for pair in tqdm.tqdm(eval_pairs):
        os.makedirs(f'./eval_pairs/pair{i+1}', exist_ok=True)
        # Load mask img
        mask = np.array(Image.open(f"./mount/total_relighting/{pair[0]}/{pair[1]}/mask_frame1.png"))
        for _, m in enumerate(candi.keys()):
            # print(m, pair)
            n_frames_dir = glob.glob(f"./mount/{m}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/n_frames=*")
            if len(n_frames_dir) > 0:
                n_frames_dir = n_frames_dir[0].split('/')[-1]
                frames = glob.glob(f"./mount/{m}/{pair[0]}/{pair[1]}/{candi[m]['post_misc']}/{n_frames_dir}/res_frame*.png")
            else: 
                frames = glob.glob(f"./mount/{m}/{pair[0]}/{pair[1]}/res_frame*.png")
            
            # print(frames)
            frames = sort_by_frame(frames)
            
            input_dat_path = frames[0]
            relit_dat_path = frames[-1]
            
            # if not os.path.exists(f'./eval_pairs/pair{i+1}/input_pair{i+1}.png'):
                # os.system(f'ln -sf {input_dat_path} ./eval_pairs/pair{i+1}/input_pair{i+1}.png')
            os.system(f'cp {input_dat_path} ./eval_pairs/pair{i+1}/input_pair{i+1}.png')
                
            # if not os.path.exists(f'./eval_pairs/pair{i+1}/{m}_pair{i+1}.png'):
                # os.system(f'ln -sf {relit_dat_path} ./eval_pairs/pair{i+1}/{m}_pair{i+1}.png')
            
            # if not os.path.exists(f'./eval_pairs/pair{i+1}/target_pair{i+1}.jpg'):
            os.system(f"cp {dataset_path}/{pair[1].split('=')[-1]} ./eval_pairs/pair{i+1}/target_pair{i+1}.jpg")
            
            gen_ball.drawSH(params[pair[1].split('=')[-1]]['light'], f"./eval_pairs/pair{i+1}/sh_pair{i+1}.jpg")
            
            # With bg
            os.system(f'cp {relit_dat_path} ./eval_pairs/pair{i+1}/{m}_pair{i+1}_bg.png')
            # Without bg
            img = np.array(Image.open(relit_dat_path))
            unmask = mask[..., None] * img
            unmask = np.clip(unmask.astype(np.uint8), 0, 255)
            Image.fromarray(unmask).save(f"./eval_pairs/pair{i+1}/{m}_pair{i+1}_nobg.png")
            
            # Facial part
            mask_anno = np.array(Image.open(f"/data/mint/DPM_Dataset/ffhq_256_with_anno/face_segment/valid/anno/anno_{pair[0].split('=')[-1].replace('.jpg', '.png')}"))
            mask_facial = face_segment('faceseg_face&noears', mask_anno)
            facial_part = mask_facial[..., None] * img
            facial_part = np.clip(facial_part.astype(np.uint8), 0, 255)
            Image.fromarray(facial_part).save(f"./eval_pairs/pair{i+1}/{m}_pair{i+1}_facial.png")
            
            non_facial_part = (~mask_facial)[..., None] * img
            non_facial_part = np.clip(non_facial_part.astype(np.uint8), 0, 255)
            Image.fromarray(non_facial_part).save(f"./eval_pairs/pair{i+1}/{m}_pair{i+1}_nonfacial.png")
            
            mask_bg = face_segment('faceseg_bg', mask_anno)
            non_facial_blur_bg = ~mask_bg[..., None] * img
            non_facial_blur_bg = np.clip(non_facial_blur_bg.astype(np.uint8), 0, 255)
            Image.fromarray(non_facial_blur_bg).save(f"./eval_pairs/pair{i+1}/{m}_pair{i+1}_nonfacial_blur_bg.png")
            
            wm = Image.open("/home/mint/Dev/DiFaReli/user_study/difareli_user_study/change_bg/dat/randomly_for_mturk/bw_overlay.jpg").convert('RGB')
            wm = np.array(wm.resize((256, 256))).astype(np.uint8)
            
            # wm = np.zeros_like(img)
            # wm[..., 0] += 255
            alpha = 0.6
            blend_img = wm_area(img, mask_facial, wm, alpha)
            Image.fromarray(blend_img).save(f"./eval_pairs/pair{i+1}/{m}_pair{i+1}_wm_facial.png")
            
            blend_img = wm_area(img, ~mask_bg, wm, alpha)
            Image.fromarray(blend_img).save(f"./eval_pairs/pair{i+1}/{m}_pair{i+1}_wm_bg.png")
            
            
        # Dump empty text file that named as the pair name
        with open(f"./eval_pairs/pair{i+1}/{pair[0]}_{pair[1]}.txt", 'w') as f:
            f.write('')
        i += 1
            
if __name__ == '__main__':
    pl.seed_everything(args.seed)
    if args.do_mount:
        mount_dat()
    if args.gen_pairs:
        gen_pairs()
