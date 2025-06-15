import os
import torch
import torchaudio
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from hubconf import ReDimNet
import tempfile


def load_audio(filepath, target_sr=16000):
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception:
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            audio.export(tmpfile.name, format="wav")
            waveform, sr = torchaudio.load(tmpfile.name)
            os.remove(tmpfile.name)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def extract_embedding(model, waveform):
    with torch.no_grad():
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        waveform = waveform[:, :16000].unsqueeze(0)
        emb = model(waveform)
        return emb.squeeze().cpu().numpy()


def group_by_gap(segments: List[Dict], max_gap=10.0) -> List[List[Dict]]:
    segments = sorted(segments, key=lambda x: x['start'])
    groups = [[segments[0]]]
    for seg in segments[1:]:
        if seg['start'] - groups[-1][-1]['end'] <= max_gap:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return groups


def cluster_within_group(group: List[Dict]) -> List[List[Dict]]:
    clusters = []
    for seg in group:
        assigned = False
        for cluster in clusters:
            sims_to_cluster = [cosine_similarity(seg['embedding'].reshape(1, -1), other['embedding'].reshape(1, -1))[0][0]
                               for other in cluster]
            cluster_valid = all(
                cosine_similarity(a['embedding'].reshape(1, -1), b['embedding'].reshape(1, -1))[0][0] >= 0.1
                for i, a in enumerate(cluster) for j, b in enumerate(cluster) if i < j)
            if all(s > 0.4 for s in sims_to_cluster) and cluster_valid:
                cluster.append(seg)
                assigned = True
                break
        if not assigned:
            clusters.append([seg])
    return clusters


def compute_clusters(segments: List[Dict], model=None, max_gap=10.0) -> List[List[List[Dict]]]:
    print("[INFO] 初始化模型...")
    if model is None:
        model = ReDimNet(model_name='b6', train_type='ptn', dataset='vox2')
        model.eval()

    print("[INFO] 提取嵌入...")
    for i, seg in enumerate(segments):
        waveform = load_audio(seg['audio_path'])
        seg['embedding'] = extract_embedding(model, waveform)
        seg['index'] = seg.get('index', i)  # 保底设置 index

    print("[INFO] 分组并聚类...")
    groups = group_by_gap(segments, max_gap=max_gap)
    all_clusters = []

    for group_id, group in enumerate(groups):
        for seg in group:
            seg['group'] = group_id  # ✅ 给每个片段打上分组编号
        clusters = cluster_within_group(group)
        all_clusters.append(clusters)

    return all_clusters