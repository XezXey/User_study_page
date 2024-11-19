from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import blobfile as bf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample_pair_json', required=True)
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
args = parser.parse_args()

path = "/home/mint/Dev/DiFaReli/user_study/difareli_user_study/change_bg/dat/randomly_for_mturk/mount/"

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

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

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        s = request.args.get('s', default=0, type=int) #NOTE: Start index
        e = request.args.get('e', default=10, type=int)
        #NOTE: Root path
        method = ["difareli", "difareli_canny=153to204", 
                  "hou21_shadowm", "hou22_geom", "total_relighting"]
        
        with open(args.sample_pair_json, 'r') as f:
            sample_pair = json.load(f)['pair']
        all_key = list(sample_pair.keys())[int(s):int(e)]
        # print(all_key, sample_pair)
        out = ""
        out += "<table>"
        for k in all_key:
            out += "<tr>"
            out += f"<td> {k}:{sample_pair[k]} </td>"
            for m in method:
                out += f"<td> {m} </td>"
            out += "</tr>"
        
            v = sample_pair[k]
            # Display images
            # Input-Target
            out += "<tr>"
            inp_f = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/{v['src']}"
            tgt_f = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/{v['dst']}"
            out += f"<td><img src='/files/{inp_f}' width='256' height='256'>"
            out += f"<img src='/files/{tgt_f}' width='256' height='256'></td>"
            
            # Relit
            
            for m in method:
                img_path = f"/home/mint/Dev/DiFaReli/user_study/difareli_user_study/change_bg/dat/randomly_for_mturk/mount/{m}/src={v['src']}/dst={v['dst']}/"
                if 'difareli' in m:
                    if len(glob.glob(f"{img_path}/Lerp_1000/n_frames=*")) <= 0:
                        continue
                    else:
                        img_path = glob.glob(f"{img_path}/Lerp_1000/n_frames=*")[0]
                        print(img_path)
                if os.path.exists(img_path):
                    frames = sort_by_frame(glob.glob(f"{img_path}/res_*.png"))
                    out += f"<td> <img src='/files/{frames[-1]}' width='256' height='256'> </td>"

            out += "</tr>"
            out += "<br>"
            
        out += "</table>"
        return out
    return app

if __name__ == "__main__":
    
    data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/"
    img_path = _list_image_files_recursively(data_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)