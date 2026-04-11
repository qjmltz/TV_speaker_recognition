import os
import yaml
from datetime import timedelta

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
def save_labeled_srt(results, output_path, actor_map):
    # 排序
    results.sort(key=lambda x: x.get("segment_id", -1))

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(results, 1):
            start_str = seconds_to_srt_time(seg['start'])
            end_str = seconds_to_srt_time(seg['end'])

            actor_name = actor_map.get(seg['actor'], seg['actor'])
            text = seg.get('text', '')

            f.write(f"{idx}\n{start_str} --> {end_str}\n{actor_name}: {text}\n\n")

    print(f"[INFO] 已保存 SRT：{output_path}")