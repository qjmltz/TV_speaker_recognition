import os
import torch
import torchaudio
import numpy as np
import collections
import tempfile
from typing import List, Dict
from hubconf import ReDimNet
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment


class VoiceClassifier:
    def __init__(self, ref_dir='dataset/ref/'):
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")

        # 初始化模型，加载预训练权重，切换评估模式
        self.model = ReDimNet(model_name='b6', train_type='ptn', dataset='vox2')
        self.model.to(self.device)  # ✅ 模型迁移到设备
        self.model.eval()

        # 构建参考库，加载参考音频并提取平均嵌入向量
        self.actor_avg_embeds = self._build_reference_library(ref_dir)

    def _load_audio(self, filepath, target_sr=16000):
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

    def _extract_embedding(self, waveform):
        with torch.no_grad():
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[1] < 16000:
                pad_len = 16000 - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            waveform = waveform[:, :16000].unsqueeze(0).to(self.device)  # ✅ 张量送到设备
            emb = self.model(waveform)
            return emb.squeeze().cpu().numpy()  # ✅ 推理后转回 CPU 再转 numpy

    def _extract_embeddings_sliding(self, waveform, window_size=16000, stride=16000):
        segments = []
        total_len = waveform.shape[1]
        for start in range(0, total_len - window_size + 1, stride):
            segment = waveform[:, start:start + window_size]
            if segment.shape[1] < window_size:
                pad_len = window_size - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, pad_len))
            segments.append(segment)
        return [self._extract_embedding(seg) for seg in segments]

    def _build_reference_library(self, ref_dir):
        actor_embeds = {}
        for fname in os.listdir(ref_dir):
            if fname.endswith('.wav'):
                actor = fname.split('_')[0]
                wav = self._load_audio(os.path.join(ref_dir, fname))
                wav = wav[:, :32000]
                emb = self._extract_embedding(wav[:, :16000])
                actor_embeds.setdefault(actor, []).append(emb)
        return {k: np.mean(v, axis=0) for k, v in actor_embeds.items()}

    def classify(self, all_clusters: List[List[Dict]]) -> List[Dict]:
        results = []

        for group_idx, group_clusters in enumerate(all_clusters, 1):
            for cluster_idx, cluster in enumerate(group_clusters, 1):
                vote_counter = collections.Counter()
                sim_sum = collections.defaultdict(float)
                sim_count = collections.defaultdict(int)

                for seg in cluster:
                    waveform = self._load_audio(seg['audio_path'])
                    total_len = waveform.shape[1]
                    seen_actors = set(seg.get('seen_actors', []))

                    if total_len < 16000:
                        emb = self._extract_embedding(waveform)
                        sims = {
                            actor: cosine_similarity(emb.reshape(1, -1), ref.reshape(1, -1))[0][0] * (
                                1.5 if actor in seen_actors else 1.0)
                            for actor, ref in self.actor_avg_embeds.items()
                        }
                        best_actor = max(sims, key=sims.get)
                        vote_counter[best_actor] += 1
                        sim_sum[best_actor] += sims[best_actor]
                        sim_count[best_actor] += 1
                    else:
                        embeddings = self._extract_embeddings_sliding(waveform)
                        for emb in embeddings:
                            sims = {
                                actor: cosine_similarity(emb.reshape(1, -1), ref.reshape(1, -1))[0][0] * (
                                    1.5 if actor in seen_actors else 1.0)
                                for actor, ref in self.actor_avg_embeds.items()
                            }
                            best_actor = max(sims, key=sims.get)
                            vote_counter[best_actor] += 1
                            sim_sum[best_actor] += sims[best_actor]
                            sim_count[best_actor] += 1

                final_actor = vote_counter.most_common(1)[0][0]
                vote_ratio = vote_counter[final_actor] / sum(vote_counter.values())
                final_sim = sim_sum[final_actor] / sim_count[final_actor]

                seg_ids = [seg['index'] for seg in cluster]
                results.append({
                    'group': group_idx,
                    'class': cluster_idx,
                    'segment_ids': seg_ids,
                    'actor': final_actor,
                    'vote_ratio': vote_ratio,
                    'similarity': final_sim,
                    'segments': cluster
                })

                print(f"[CLUSTER] Group {group_idx} Class {cluster_idx} -> {final_actor} "
                      f"(vote: {vote_ratio:.2%}, sim: {final_sim:.4f})")

        return results