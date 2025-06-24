import torch
import os
import numpy as np
from face_alignment import align
from face_alignment.net import build_model
from PIL import Image
from typing import List, Dict
import cv2

adaface_models = {
    'ir_50': "pretrained/adaface_ir50_webface4m.ckpt",
}

# 自动选择设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_model(architecture='ir_50'):
    assert architecture in adaface_models
    model = build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location=DEVICE)['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval().to(DEVICE)
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
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
            actor_id = fname.split('_')[0]
            img_path = os.path.join(ref_dir, fname)
            aligned = align.get_aligned_face(img_path)
            if aligned is None:
                continue
            tensor = to_input(aligned).to(DEVICE)
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

    grouped_segments = {}
    for seg in segments:
        grouped_segments.setdefault(seg['group'], []).append(seg)

    for group_id, group_segs in grouped_segments.items():
        start = min(seg['start'] for seg in group_segs)
        end = max(seg['end'] for seg in group_segs)
        duration = end - start
        if duration <= 0:
            for seg in group_segs:
                seg['seen_actors'] = {}
            continue

        frame_times = np.linspace(start, end, num=min(int(duration * 5), 20))
        aligned_faces = []
        valid_times = []
        for t in frame_times:
            frame_img = extract_frame_at_time(video_path, t)
            if frame_img is None:
                continue
            aligned_face = align.get_aligned_face(frame_img)
            if aligned_face is not None:
                aligned_faces.append(to_input(aligned_face))
                valid_times.append(t)

        if not aligned_faces:
            for seg in group_segs:
                seg['seen_actors'] = {}
            print(f"\n[Group {group_id}] Time: {start:.2f}s → {end:.2f}s\n  No face recognized.")
            continue

        # 打包 batch
        batch_tensor = torch.cat(aligned_faces, dim=0).to(DEVICE)  # shape: [N, 3, H, W]
        with torch.no_grad():
            feats, _ = model(batch_tensor)

        seen_actors = {}
        remaining_actors = set(ref_embeddings.keys())
        for i in range(feats.shape[0]):
            emb = feats[i].unsqueeze(0)
            for actor in list(remaining_actors):
                ref_emb = ref_embeddings[actor]
                sim = torch.nn.functional.cosine_similarity(emb, ref_emb).item()
                if sim >= threshold:
                    seen_actors[actor] = sim
                    remaining_actors.remove(actor)

        print(f"\n[Group {group_id}] Time: {start:.2f}s → {end:.2f}s")
        if seen_actors:
            for actor, score in seen_actors.items():
                print(f"  Actor {actor}: Max Similarity = {score:.4f}")
        else:
            print("  No face recognized.")

        for seg in group_segs:
            seg['seen_actors'] = seen_actors.copy()

    return segments

