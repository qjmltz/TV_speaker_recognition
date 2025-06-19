import os
import yaml
from pathlib import Path
from datetime import timedelta

from preprocessor import extract_segments_from_video
from classifier import VoiceClassifier
from clusterer import compute_clusters
from actorfaces import recognize_faces_in_segments
# 读取 actor.yaml 的角色映射
def load_actor_mapping(yaml_path: str) -> dict:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        return data.get("Actor", data)  # 支持顶层或嵌套结构

# 秒转为 SRT 时间格式
def seconds_to_srt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# 写入带角色标注的 SRT 字幕
def save_labeled_srt(results, output_path: str, actor_map: dict):
    # 1. 整理所有带 actor 的片段
    all_segs = []
    for res in results:
        actor = res['actor']
        seg_ids = res.get('segment_ids', [])  # 原始字幕编号
        for i, seg in enumerate(res['segments']):
            seg = seg.copy()
            seg['actor'] = actor
            seg['segment_id'] = seg_ids[i] if i < len(seg_ids) else -1  # 防止越界
            all_segs.append(seg)

    # 2. 按 segment_id 升序排序
    all_segs.sort(key=lambda x: x.get('segment_id', -1))

    # 3. 写入到 SRT 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(all_segs, 1):
            start_str = seconds_to_srt_time(seg['start'])
            end_str = seconds_to_srt_time(seg['end'])
            actor_name = actor_map.get(seg['actor'], seg['actor'])
            text = seg.get('text', '')
            f.write(f"{idx}\n{start_str} --> {end_str}\n{actor_name}: {text}\n\n")

    print(f"[INFO] 已保存带标注字幕至：{output_path}")


def main():
    mp4_path = "S01E02.mp4"
    srt_path = "S01E02.srt"
    yaml_path = "actor.yaml"
    output_srt_path = "S01E02_labeled.srt"

    assert os.path.exists(mp4_path), f"找不到视频文件 {mp4_path}"
    assert os.path.exists(srt_path), f"找不到字幕文件 {srt_path}"
    assert os.path.exists(yaml_path), f"找不到角色映射文件 {yaml_path}"

    # 加载角色映射
    actor_map = load_actor_mapping(yaml_path)

    # 打印映射确认
    print("[INFO] 角色映射如下：")
    for key, name in actor_map.items():
        print(f"  {key} -> {name}")

    # 步骤 1：提取字幕音频片段
    print("[STEP 1] 提取字幕音频片段")
    segments = extract_segments_from_video(mp4_path, srt_path)
    print("[STEP 2] 演员人脸识别片段")
    segments = recognize_faces_in_segments(mp4_path, segments, ref_dir='dataset/ref')
    # 步骤 2：同组聚类
    print("[STEP 3] 上下文音频聚类")
    all_clusters = compute_clusters(segments)
    # 步骤 2：语音角色分类
    print("[STEP 4] 分类识别角色")

    classifier = VoiceClassifier(ref_dir='dataset/ref/')
    results = classifier.classify(all_clusters)
    for res in results:
        if res.get("similarity", 1.0) < 0.2:
            res['actor'] = "其他"
    # 步骤 3：写入带标注字幕
    print("[STEP 3] 写入标注字幕")
    save_labeled_srt(results, output_srt_path, actor_map)

if __name__ == "__main__":
    main()