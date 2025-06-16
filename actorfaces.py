import torch
import os
import numpy as np
from face_alignment import align
from face_alignment.net import build_model
from PIL import Image
from typing import List, Dict
import cv2

adaface_models = {
    'ir_50':"pretrained/adaface_ir50_webface4m.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    assert architecture in adaface_models
    model = build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    array = np.array([brg_img.transpose(2, 0, 1)], dtype=np.float32)
    return torch.from_numpy(array)

def extract_frame_at_time(video_path: str, time_sec: float):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def load_reference_embeddings(ref_dir: str, model):
    embeddings = {}
    for fname in os.listdir(ref_dir):
        if fname.endswith('.jpg'):
            actor_id = fname.split('_')[0]  # A_0.jpg -> A

            img_path = os.path.join(ref_dir, fname)
            aligned = align.get_aligned_face(img_path)
            if aligned is None:
                continue
            tensor = to_input(aligned)
            with torch.no_grad():
                feat, _ = model(tensor)
            if actor_id in embeddings:
                embeddings[actor_id].append(feat)
            else:
                embeddings[actor_id] = [feat]
    return {k: torch.mean(torch.cat(v, dim=0), dim=0, keepdim=True) for k, v in embeddings.items()}

def recognize_faces_in_segments(video_path: str, segments: List[Dict], ref_dir: str, threshold=0.4) -> List[Dict]:
    model = load_pretrained_model('ir_50')
    ref_embeddings = load_reference_embeddings(ref_dir, model)

    # 根据 group 编号对 segments 分组
    grouped_segments = {}
    for seg in segments:
        grouped_segments.setdefault(seg['group'], []).append(seg)

    for group_id, group_segs in grouped_segments.items():
        start = min(seg['start'] for seg in group_segs)
        end = max(seg['end'] for seg in group_segs)
        duration = end - start
        if duration <= 0:
            for seg in group_segs:
                seg['seen_actors'] = {}  # 空字典
            continue

        frame_times = np.linspace(start, end, num=min(int(duration * 5), 20))
        seen_actors = {}  # actor_id -> max_sim
        remaining_actors = set(ref_embeddings.keys())  # 每个 group 只识别一次

        for t in frame_times:
            if not remaining_actors:
                break  # 所有人都识别过了，不需要继续识别

            frame_img = extract_frame_at_time(video_path, t)
            if frame_img is None:
                continue
            aligned_face = align.get_aligned_face(frame_img)
            if aligned_face is None:
                continue

            input_tensor = to_input(aligned_face)
            with torch.no_grad():
                emb, _ = model(input_tensor)

            for actor in list(remaining_actors):
                ref_emb = ref_embeddings[actor]
                sim = torch.nn.functional.cosine_similarity(emb, ref_emb).item()
                if sim >= threshold:
                    seen_actors[actor] = sim
                    remaining_actors.remove(actor)  # 一旦匹配成功就移除

        print(f"\n[Group {group_id}] Time: {start:.2f}s → {end:.2f}s")
        if seen_actors:
            for actor, score in seen_actors.items():
                print(f"  Actor {actor}: Max Similarity = {score:.4f}")
        else:
            print("  No face recognized.")

        # 将识别结果写入该 group 内的每个 segment
        for seg in group_segs:
            seg['seen_actors'] = seen_actors.copy()  # 每段都有这个字段

    return segments

