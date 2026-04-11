import json
import os
from collections import defaultdict
from openai import OpenAI


import json
import os
import yaml
from collections import defaultdict
from openai import OpenAI


class LLMRefiner:
    def __init__(self, api_key=None, actor_yaml="actor.yaml"):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        # ✅ 加载角色映射
        self.actor_map = self.load_actor_mapping(actor_yaml)
        self.inv_actor_map = {v: k for k, v in self.actor_map.items()}

    def load_actor_mapping(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get("Actor", data)

    # =========================
    # 1. 按 scene(group) 聚合
    # =========================
    def build_scenes(self, results):
        scenes = defaultdict(lambda: {
            "group": None,
            "face_set": set(),
            "dialogues": [],
            "raw_indices": []  # 👉 对应原results位置
        })

        for idx, item in enumerate(results):
            g = item["group"]
            scenes[g]["group"] = g

            # 人脸角色（转中文）
            seen = item.get("seen_actors", {})
            scenes[g]["face_set"].update(
                [self.actor_map.get(k, k) for k in seen.keys()]
            )

            # 对话
            scenes[g]["dialogues"].append({
                "text": item["text"],
                "pred_actor": self.actor_map.get(item["actor"], item["actor"]),
                "similarity": item["similarity"]
            })

            scenes[g]["raw_indices"].append(idx)

        return list(scenes.values())

    # =========================
    # 2. Prompt（强化约束）
    # =========================
    def build_prompt(self, scene):
        dialogue_str = "\n".join([
            f"{i}: ({d['pred_actor']}, {d['similarity']:.2f}) {d['text']}"
            for i, d in enumerate(scene["dialogues"])
        ])

        prompt = f"""
你是一个电视剧字幕说话人纠错助手。

当前是一个场景：
出现角色（人脸识别）：{scene['face_set']}

规则：
1. 对话必须合理（问答不能同一人）
2. 第一人称要一致
3. 优先使用出现过的人脸角色
4. similarity低的更容易错
5. 不要过度修改
6.必须严格输出JSON：只输出JSON数组 不要解释 不要```json 不要任何额外内容

格式：
[
  {{"index": 0, "actor": "角色名"}}
]

台词：
{dialogue_str}
"""
        return prompt

    # =========================
    # 3. 清洗LLM输出（关键）
    # =========================
    def clean_llm_output(self, content):
        content = content.strip()

        # 去 ```json
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])

        # 截取 JSON 区间
        l = content.find("[")
        r = content.rfind("]")
        if l != -1 and r != -1:
            content = content[l:r+1]

        return content

    # =========================
    # 4. 单scene处理
    # =========================
    def refine_scene(self, scene, results):
        prompt = self.build_prompt(scene)

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是字幕纠错助手"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            content = response.choices[0].message.content
            print(f"[DEBUG RAW]\n{content}\n")

            content = self.clean_llm_output(content)

            print(f"[DEBUG CLEAN]\n{content}\n")

            corrections = json.loads(content)

            # 👉 写回原 results
            for item in corrections:
                idx = item["index"]
                if 0 <= idx < len(scene["raw_indices"]):
                    raw_idx = scene["raw_indices"][idx]

                    actor_name = item["actor"]

                    # 中文 → ID
                    actor_id = self.inv_actor_map.get(actor_name, actor_name)

                    results[raw_idx]["actor"] = actor_id

        except Exception as e:
            print("[LLM ERROR]", e)

        return results

    # =========================
    # 5. 全量处理（保持结构）
    # =========================
    def refine_all(self, results):
        scenes = self.build_scenes(results)

        print(f"[INFO] 共 {len(scenes)} 个场景")

        for scene in scenes:
            print(f"[LLM] Processing Scene {scene['group']}")
            results = self.refine_scene(scene, results)

        return results



# =========================
# 简易 main（调试用）
# =========================

from util import load_actor_mapping, save_labeled_srt

if __name__ == "__main__":
    input_path = "results.json"
    output_srt_path = "S01E01_1.srt"
    yaml_path = "actor.yaml"
    # 加载角色映射
    actor_map = load_actor_mapping(yaml_path)
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    refiner = LLMRefiner(api_key="your_key_id" ,actor_yaml="actor.yaml")

    results = refiner.refine_all(results)
    for res in results:
        if res.get("similarity", 1.0) < 0.2:
            res['actor'] = "其他"
    # 步骤 5：写入带标注字幕
    print("[STEP 5] 写入标注字幕")
    save_labeled_srt(results, output_srt_path, actor_map)





