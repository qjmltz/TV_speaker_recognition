import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def extract_wav_from_mp4(mp4_path, wav_path, sample_rate=16000):
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(wav_path, fps=sample_rate, codec='pcm_s16le')
    voice_enhancer(wav_path, wav_path)  # 调用音频增强函数


def srt_time_to_seconds(tstr):
    h, m, rest = tstr.split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(srt_path) -> List[Dict]:
    dialogues = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            time_line = lines[1]
            start_str, end_str = time_line.split(' --> ')
            start_sec = srt_time_to_seconds(start_str)
            end_sec = srt_time_to_seconds(end_str)
            text = " ".join(lines[2:])
            dialogues.append({
                'start': start_sec,
                'end': end_sec,
                'text': text
            })
    return dialogues


def group_dialogues_by_interval(dialogues: List[Dict], max_gap: float = 10.0) -> List[List[Dict]]:
    """
    将字幕按照时间间隔分组，前后句间隔超过 max_gap 秒就分组。
    """
    if not dialogues:
        return []

    groups = []
    current_group = [dialogues[0]]
    for i in range(1, len(dialogues)):
        prev = dialogues[i - 1]
        curr = dialogues[i]
        gap = curr['start'] - prev['end']
        if gap <= max_gap:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    if current_group:
        groups.append(current_group)
    return groups

def voice_enhancer(audio_path = '', output_path=''):
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model='iic/speech_zipenhancer_ans_multiloss_16k_base')
    result = ans(
        audio_path,
    output_path=output_path)
    print("done")

def extract_segments_from_video(mp4_path: str, srt_path: str) -> List[Dict]:
    """
    提取音频段，返回包含每段对应的音频路径、时间戳、字幕的列表。
    如果 save_segments=True，音频片段会保存到 dataset/query/line_XXXX.wav。
    """
    base_dir = Path(".")
    wav_path = base_dir / "temp_full.wav"
    query_dir = base_dir / "dataset/query"
    ref_dir = base_dir / "dataset/ref"


    if not query_dir.exists():
        query_dir.mkdir(parents=True, exist_ok=True)

    if not ref_dir.exists():
        raise FileNotFoundError("[ERROR] 缺少参考库文件夹：dataset/ref，请添加参考音频")

    print(f"[INFO] 提取音频：{mp4_path}")
    extract_wav_from_mp4(mp4_path, str(wav_path))
    audio = AudioSegment.from_wav(str(wav_path))
    dialogues = parse_srt(srt_path)
    dialogue_groups = group_dialogues_by_interval(dialogues)

    segments = []
    index = 1

    print(f"[INFO] 切割音频，共分为 {len(dialogue_groups)} 组...")
    for group_id, group in enumerate(dialogue_groups, 1):
        for dlg in group:
            segment = audio[dlg['start'] * 1000 : dlg['end'] * 1000]
            out_path = query_dir / f"line_{index:04d}.wav"


            segment.export(str(out_path), format='wav')

            segments.append({
                'index': index,
                'group': group_id,
                'start': dlg['start'],
                'end': dlg['end'],
                'text': dlg['text'],
                'audio_path': str(out_path)
            })
            index += 1

    os.remove(str(wav_path))
    print(f"[INFO] 完成预处理，共切割 {len(segments)} 段，对话分组数量：{len(dialogue_groups)}。")
    return segments