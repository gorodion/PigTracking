import torch
import cv2
import h5py as h5
import os.path as fs
from torch.utils.data import DataLoader
import albumentations as albu
from torchvision import transforms
import random
import numpy as np
import sys

from config import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DETECTOR = torch.hub.load('ultralytics/yolov5', 'custom', path=DETECTOR_PATH)
SEGMENTATOR = torch.load(SEGMENTATOR_PATH, map_location=DEVICE).eval()


def detect(frame):
    results = DETECTOR([frame])
    predicts = results.xyxy[0]
    predicts = predicts[predicts[:, 4] > CONF].cpu().numpy()
    return predicts

def extract_crops(frame, bboxes):
    crops = []
    for x0, y0, x1, y1 in bboxes:
        crops.append(frame[y0:y1, x0:x1])
    return crops



def make_dloader(crops):
    test_transforms = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    crop_tensors = [test_transforms(cv2.resize(crop, SIZES)) for crop in crops]
    dl = DataLoader(crop_tensors, batch_size=8, num_workers=2)
    return dl

@torch.no_grad()
def get_segmentations(dl):
    segmentations = []
    for x in dl:
        x = x.to(DEVICE)
        segm = SEGMENTATOR(x)
        segm = (segm['out'].squeeze(1).cpu().numpy() > 0).astype('uint8')
        segmentations.extend(segm)
    return segmentations


def predict_frame(frame):
    predicts = detect(frame)
    bboxes = predicts[:, :4].astype(int)
    crops = extract_crops(frame, bboxes)
    dl = make_dloader(crops)
    segmentations = get_segmentations(dl)

    assert len(crops) == len(segmentations)

    segmentations = [cv2.resize(segm, crop.shape[:2][::-1]) 
        for segm, crop in zip(segmentations, crops)]

    return predicts, bboxes, segmentations
    

def generate_color(pigs_colors):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    while color in pigs_colors.values():
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color


def annot_frame(frame: np.array, ids, bboxes, masks, activities, pigs_colors) -> np.array:
    pigs_activity = 0
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    font_color = (255,)*3

    for id, (x0, y0, x1, y1), mask, activity in zip(ids, bboxes, masks, activities):
        color = pigs_colors.get(id)
        pigs_activity += activity

        if color is None:
            color = generate_color(pigs_colors)
            pigs_colors[id] = color

        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 4)

        zeros = np.zeros(frame.shape[:2])
        zeros[y0:y1, x0:x1] = mask
        frame[zeros == 1] = transparency * frame[zeros == 1] \
            + (1 - transparency) * np.array(color, 'uint8')

        cv2.putText(frame, f"ID: {id}. Активность: {activity:.2f}", (x0 + 2, y0 - 4), font, font_scale, color = font_color, thickness=2)

    cv2.putText(frame, 
                f"Количество: {len(bboxes)}", 
                (10, 30), 
                font, 
                font_scale, 
                color = font_color, 
                thickness=2)
    cv2.putText(frame, 
                f"Средняя активность: {pigs_activity / len(bboxes):.2f}",
                (10, 60), 
                font, 
                font_scale, 
                color = font_color, 
                thickness=2)
    return frame



def savePredict(Path, Name, boxs, masks, num_pigs_per_movie):
	ff = h5.File(fs.join(Path, Name), 'w')
	num_frames = len(boxs) # количество фреймов
	for frame_iter in range(num_frames):
		grp = ff.create_group("Frame_%d"%frame_iter)
		grp.create_dataset('boxs', data = boxs[frame_iter])
		grp.create_dataset('num_pigs', data = num_pigs_per_movie[frame_iter])
		num_pigs = len(boxs[frame_iter])
		subgrp = grp.create_group("PolyMasks")
		for pig_inter in range(num_pigs):
			subgrp.create_dataset('polymask_%d'%pig_inter, data = masks[frame_iter][pig_inter])
	ff.close()
	return None
    

def center_distance(bbox1, bbox2):
    xc1 = (bbox1[0] + bbox1[2]) / 2
    yc1 = (bbox1[1] + bbox1[3]) / 2
    xc2 = (bbox2[0] + bbox2[2]) / 2
    yc2 = (bbox2[1] + bbox2[3]) / 2
    return ((yc2 - yc1) ** 2 + (xc2 - xc1) ** 2) ** 0.5


def euclid_distance(bbox1, bbox2):
    return ((bbox2 - bbox1) ** 2).sum() ** 0.5

def find_matches(ids_bboxes, bboxes_now):
    distances = []
    for id_prev, bbox_prev in ids_bboxes.items():
        for id_now, bbox_now in enumerate(bboxes_now):
            dist = euclid_distance(bbox_prev, bbox_now)
            if dist < K:
                distances.append((dist, id_prev, id_now))
    
    distances.sort()
    ids_now = [None] * len(bboxes_now)
    ids_matched = set()
    for dist, id_prev, id_now in distances:
        if ids_now[id_now] is not None: # новый бокс уже сматчился
            pass
        elif id_prev in ids_matched: # старый бокс уже сматчился
            pass
        else:
            ids_now[id_now] = id_prev
            ids_matched.add(id_prev)

    # если из новых остались старые
    max_prev_id = max(ids_bboxes) + 1
    for n_box, id_now in enumerate(ids_now):
        if id_now is None:
            ids_now[n_box] = max_prev_id
            max_prev_id += 1
    
    return ids_now

def process_video(vid_path: str, out_dir: str):
    vid = cv2.VideoCapture(vid_path)
    ids_bboxes = {}
    out = None
    ids_distances = {}
    pigs_colors = {}
    pigs_bbox = {}

    boxs_all = []
    masks_all = []
    num_pigs_all = []
    
    while True:

        ret, orig_frame = vid.read()
        if not ret:
            break

        frame = orig_frame[..., ::-1]

        if out is None:
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            h, w = orig_frame.shape[:2]
            out = cv2.VideoWriter(os.path.join(out_dir, 'outpy.mkv'),cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h)) # TODO path

        predicts, bboxes, segmentations = predict_frame(frame)

        if len(bboxes) == 0:
            continue

        if ids_bboxes:
            now_ids = find_matches(ids_bboxes, bboxes)
        else:     
            for id, bbox in enumerate(bboxes, 1):
                ids_bboxes[id] = bbox
                ids_distances[id] = 0
            ids_bboxes = {id: bbox for id, bbox in enumerate(bboxes, 1)}
            # TODO write video
            continue

        activities = []
        for id_now, id_prev in enumerate(now_ids):
            if id_prev in ids_bboxes:
                dist = center_distance(ids_bboxes[id_prev], bboxes[id_now])
                ids_bboxes[id_prev] = bboxes[id_now]
                activity = 1 if dist > L else 0
                ids_distances[id_prev] = 0.8 * ids_distances[id_prev] + 0.2 * activity
            else:
                ids_bboxes[id_prev] = bboxes[id_now]
                ids_distances[id_prev] = 0
            activities.append(ids_distances[id_prev])

        annot_frame(orig_frame, now_ids, bboxes, segmentations, activities, pigs_colors=pigs_colors)
        out.write(orig_frame)

        boxs_all.append(bboxes)
        masks_all.append(segmentations)
        num_pigs_all.append(len(bboxes))
        
    out.release()
    savePredict(out_dir, 'out.hdf5', boxs_all, masks_all, num_pigs_all) # TODO

from pathlib import Path    
import os.path

def mainflow(dirname, out_dir='prediction'):
    for path in Path(dirname).glob('*mkv'):
        out_dir = os.path.join(out_dir, path.stem)
        os.makedirs(out_dir, exist_ok=True)
        process_video(str(path), out_dir)
        
        
if __name__ == '__main__':
    mainflow(sys.argv[1])