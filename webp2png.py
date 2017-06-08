#!/usr/bin/python
import os, sys

webp_root = "/home/laoreja/dataset/SceneFlow/frames_cleanpass_webp"
png_root = "/home/laoreja/dataset/SceneFlow/frames_cleanpass_png"

if not os.path.exists(png_root):
    os.makedirs(png_root)

def recursive(root, dest_path):
    contents = os.listdir(root)
    if contents[0].endswith('.webp'):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            
        for img_name in contents:
            webp_path = os.path.join(root, img_name)
            png_path = os.path.join(dest_path, os.path.splitext(img_name)[0]+'.png')
            os.system('dwebp '+webp_path+' -o '+png_path+' -v')
        
        return
    else:
        for subdir in contents:
            recursive(os.path.join(root, subdir), os.path.join(dest_path, subdir))
        
recursive(webp_root, png_root)
