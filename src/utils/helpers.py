"""
The methods declared in this file were inspired or extracted from the following code:

https://github.com/antgroup/echomimic_v2/blob/main/demo.ipynb

which is Licensed under the Apache License, Version 2.0.

"""

import sys
from utils.img_utils import (
    pil_to_cv2,
    cv2_to_pil,
    center_crop_cv2,
    pils_from_video,
    save_videos_from_pils,
    save_video_from_cv2_list,
)
from PIL import Image
import cv2
from IPython import embed
import numpy as np
import copy
from utils.motion_utils import motion_sync
import pathlib
import torch
import pickle
from glob import glob
import os
from models.dwpose.dwpose_detector import DWposeDetector
from models.dwpose.util import draw_pose
import decord
from tqdm import tqdm
from moviepy.editor import AudioFileClip, VideoFileClip
from multiprocessing.pool import ThreadPool


def convert_fps(src_path, tgt_path, tgt_fps=24, tgt_sr=16000):
    clip = VideoFileClip(src_path)
    new_clip = clip.set_fps(tgt_fps)
    if tgt_fps is not None:
        audio = new_clip.audio
        audio = audio.set_fps(tgt_sr)
        new_clip = new_clip.set_audio(audio)
    if ".mov" in tgt_path:
        tgt_path = tgt_path.replace(".mov", ".mp4")
    new_clip.write_videofile(tgt_path, codec="libx264", audio_codec="aac")


def get_video_pose(
    video_path: str,
    model_det_path: str,
    model_pose_path: str,
    device: str = "cuda",
    sample_stride: int = 1,
    max_frame=None,
):
    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    # print(frames[0])
    if max_frame is not None:
        frames = frames[0:max_frame, :, :]
    height, width, _ = frames[0].shape

    dwpose_detector = DWposeDetector(
        model_det=model_det_path,
        model_pose=model_pose_path,
        device=device,
    )

    detected_poses = [dwpose_detector(frm) for frm in frames]
    dwpose_detector.release_memory()

    return detected_poses, height, width, frames


def resize_and_pad(img, max_size):
    img_new = np.zeros((max_size, max_size, 3)).astype("uint8")
    imh, imw = img.shape[0], img.shape[1]
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw / imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half - half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh / imw * imw_new))
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half - half_h
        re = rb + imh_new

    img_resize = cv2.resize(img, (imw_new, imh_new))
    img_new[rb:re, cb:ce, :] = img_resize
    return img_new


def resize_and_pad_param(imh, imw, max_size):
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw / imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half - half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh / imw * imw_new))
        imh_new = max_size

        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half - half_h
        re = rb + imh_new

    return imh_new, imw_new, rb, re, cb, ce


def get_pose_params(detected_poses, height, width, max_size):
    # pose rescale
    w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
    mid_all = []
    for num, detected_pose in enumerate(detected_poses):
        detected_poses[num]["num"] = num
        candidate_body = detected_pose["bodies"]["candidate"]
        score_body = detected_pose["bodies"]["score"]
        candidate_face = detected_pose["faces"]
        score_face = detected_pose["faces_score"]
        candidate_hand = detected_pose["hands"]
        score_hand = detected_pose["hands_score"]

        # face
        if candidate_face.shape[0] > 1:
            index = 0
            candidate_face = candidate_face[index]
            score_face = score_face[index]
            detected_poses[num]["faces"] = candidate_face.reshape(
                1, candidate_face.shape[0], candidate_face.shape[1]
            )
            detected_poses[num]["faces_score"] = score_face.reshape(
                1, score_face.shape[0]
            )
        else:
            candidate_face = candidate_face[0]
            score_face = score_face[0]

        # body
        if score_body.shape[0] > 1:
            tmp_score = []
            for k in range(0, score_body.shape[0]):
                tmp_score.append(score_body[k].mean())
            index = np.argmax(tmp_score)
            candidate_body = candidate_body[index * 18 : (index + 1) * 18, :]
            score_body = score_body[index]
            score_hand = score_hand[(index * 2) : (index * 2 + 2), :]
            candidate_hand = candidate_hand[(index * 2) : (index * 2 + 2), :, :]
        else:
            score_body = score_body[0]
        all_pose = np.concatenate((candidate_body, candidate_face))
        all_score = np.concatenate((score_body, score_face))
        all_pose = all_pose[all_score > 0.8]

        body_pose = np.concatenate((candidate_body,))
        mid_ = body_pose[1, 0]

        face_pose = candidate_face
        hand_pose = candidate_hand

        h_min, h_max = np.min(face_pose[:, 1]), np.max(body_pose[:7, 1])

        h_ = h_max - h_min

        mid_w = mid_
        w_min = mid_w - h_ // 2
        w_max = mid_w + h_ // 2

        w_min_all.append(w_min)
        w_max_all.append(w_max)
        h_min_all.append(h_min)
        h_max_all.append(h_max)
        mid_all.append(mid_w)

    w_min = np.min(w_min_all)
    w_max = np.max(w_max_all)
    h_min = np.min(h_min_all)
    h_max = np.max(h_max_all)
    mid = np.mean(mid_all)

    margin_ratio = 0.25
    h_margin = (h_max - h_min) * margin_ratio

    h_min = max(h_min - h_margin * 0.65, 0)
    h_max = min(h_max + h_margin * 0.05, 1)

    h_new = h_max - h_min

    h_min_real = int(h_min * height)
    h_max_real = int(h_max * height)
    mid_real = int(mid * width)

    height_new = h_max_real - h_min_real + 1
    width_new = height_new
    w_min_real = mid_real - width_new // 2
    if w_min_real < 0:
        w_min_real = 0
        width_new = mid_real * 2

    w_max_real = w_min_real + width_new
    w_min = w_min_real / width
    w_max = w_max_real / width

    imh_new, imw_new, rb, re, cb, ce = resize_and_pad_param(
        height_new, width_new, max_size
    )
    res = {
        "draw_pose_params": [imh_new, imw_new, rb, re, cb, ce],
        "pose_params": [w_min, w_max, h_min, h_max],
        "video_params": [h_min_real, h_max_real, w_min_real, w_max_real],
    }
    return res


def save_pose_params_item(input_items):
    detected_pose, pose_params, draw_pose_params, save_dir = input_items
    w_min, w_max, h_min, h_max = pose_params
    num = detected_pose["num"]
    candidate_body = detected_pose["bodies"]["candidate"]
    candidate_face = detected_pose["faces"][0]
    candidate_hand = detected_pose["hands"]
    candidate_body[:, 0] = (candidate_body[:, 0] - w_min) / (w_max - w_min)
    candidate_body[:, 1] = (candidate_body[:, 1] - h_min) / (h_max - h_min)
    candidate_face[:, 0] = (candidate_face[:, 0] - w_min) / (w_max - w_min)
    candidate_face[:, 1] = (candidate_face[:, 1] - h_min) / (h_max - h_min)
    candidate_hand[:, :, 0] = (candidate_hand[:, :, 0] - w_min) / (w_max - w_min)
    candidate_hand[:, :, 1] = (candidate_hand[:, :, 1] - h_min) / (h_max - h_min)
    detected_pose["bodies"]["candidate"] = candidate_body
    detected_pose["faces"] = candidate_face.reshape(
        1, candidate_face.shape[0], candidate_face.shape[1]
    )
    detected_pose["hands"] = candidate_hand
    detected_pose["draw_pose_params"] = draw_pose_params
    np.save(save_dir + "/" + str(num) + ".npy", detected_pose)


def save_pose_params(detected_poses, pose_params, draw_pose_params, ori_video_path):
    save_dir = ori_video_path.replace("video", "pose/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_list = []

    for i, detected_pose in enumerate(detected_poses):
        input_list.append([detected_pose, pose_params, draw_pose_params, save_dir])

    pool = ThreadPool(8)
    pool.map(save_pose_params_item, input_list)
    pool.close()
    pool.join()
    return save_dir


def resize_img(frame: np.ndarray, max_size: int = 768):
    height, width, _ = frame.shape
    short_size = min(height, width)
    resize_ratio = max(max_size / short_size, 1.0)
    return cv2.resize(frame, (int(resize_ratio * width), int(resize_ratio * height)))


def get_aligned_img(ori_frame: np.ndarray, video_params: list[int], max_size: int):
    h_min_real, h_max_real, w_min_real, w_max_real = video_params
    img = ori_frame[h_min_real:h_max_real, w_min_real:w_max_real, :]
    return resize_and_pad(img, max_size=max_size)
