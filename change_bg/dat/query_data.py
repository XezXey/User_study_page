import numpy as np
import glob, os, json

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

with open('./candidates.json', 'r') as f:
    candidates = json.load(f)['pair']

for kp, p in candidates.items():
    src = p['src']
    dst = p['dst']
    sampling_path = p['sampling_path']
    postfix_path = p['postfix_path']
    input_path = p['input']['path']
    input_set = p['input']['set']
    
    os.makedirs(f'./{kp}', exist_ok=True)
    # Link the source image
    # os.system(f'ln -sf {input_path}/{input_set}/{src} ./{kp}/input_{kp}.png')
    os.system(f"ln -sf {input_path}/{input_set}/{src.replace('jpg', 'png')} ./{kp}/input_{kp}.png")
    
    # Candidates
    for kc, c in p['candidates'].items():
        frames = glob.glob(f'/{sampling_path}/{c}/src={src}/dst={dst}/{postfix_path}/res_frame*.png')
        frames = sort_by_frame(frames)
        src_outpath = frames[-1]
        dst_outpath = f'./{kp}/{kc}_{kp}.png'
        os.system(f'ln -sf {src_outpath} {dst_outpath}')
        
        
        # Create videos if not exists
        vid_outpath = f'/{sampling_path}/{c}/src={src}/dst={dst}/{postfix_path}/res_frame_rt.mp4'
        dst_outpath = f'./{kp}/{kc}_{kp}_vidrt.mp4'
        if os.path.exists(vid_outpath):
            os.system(f'ln -sf {vid_outpath} {dst_outpath}')
        else: 
            frames = frames + frames[::-1]
            with open(f'./{kp}/input_frames.txt', 'w') as f:
                for frame in frames:
                    f.write(f"file {frame}\n")
            # print(f'ffmpeg -y -r 1 {frames_txt} -preset veryslow -c:v libx264 -crf 18 {dst_outpath}')
            # exit()
            os.system(f'ffmpeg -r 25 -y -f concat -safe 0 -i ./{kp}/input_frames.txt -preset veryslow -c:v libx264 -crf 18 {dst_outpath}')
