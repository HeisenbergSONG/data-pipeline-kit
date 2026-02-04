import json
import uuid
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Any
import random
from tqdm import tqdm

# LLM clients
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from transformers import AutoTokenizer
import tiktoken  # 作为备选近似

# ==================== 配置区 ====================
OPENAI_API_KEY = "sk-xxx"
ANTHROPIC_API_KEY = "sk-ant-xxx"
GEMINI_API_KEY = "AIzaxxx"

genai.configure(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  # 最接近 Qwen3

TASK_TYPES = ["文本摘要", "阅读理解", "多文本问答", "对话补全"]
THEMES = ["新闻报道", "法律", "小说情节推理", "其他文本"]

TARGET_TOTAL = 50000
TARGET_PER_TASK = TARGET_TOTAL // 4
TARGET_CN = TARGET_TOTAL // 2
TARGET_EN = TARGET_TOTAL // 2

# 存储已生成数据（用于去重）
generated_set = set()
data_list: List[Dict] = []

# MinHash LSH 去重（阈值 0.85）
from datasketch import MinHash, MinHashLSH
lsh = MinHashLSH(threshold=0.85, num_perm=128)

def get_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=128)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# ==================== 幻觉检测模板 ====================
HALLUCINATION_PROMPT = """
Your task is to analyze a response to a user query and identify any instances of hallucinations...
{template here, copy from SOW}
"""

def check_hallucination(context: str, question: str, answer: str) -> Dict:
    full_context = f"Context: {context}\nQuestion: {question}"
    results = {}
    models = ["gpt-4o", "gemini-1.5-flash", "claude-3-5-sonnet-20240620"]
    
    for model_name in models:
        if "gpt" in model_name:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": HALLUCINATION_PROMPT.format(context=full_context, response=answer)}],
                temperature=0
            )
            result = json.loads(resp.choices[0].message.content)
        elif "gemini" in model_name:
            resp = gemini_model.generate_content(HALLUCINATION_PROMPT.format(context=full_context, response=answer))
            result = json.loads(resp.text)
        else:  # claude
            msg = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                messages=[{"role": "user", "content": HALLUCINATION_PROMPT.format(context=full_context, response=answer)}]
            )
            result = json.loads(msg.content[0].text)
        
        results[model_name] = result
    
    # 至少2个模型无幻觉
    no_halluc_count = sum(1 for r in results.values() if r["final_result"]["conflict"] == 0 and r["final_result"]["baseless_info"] == 0)
    return {"results": results, "pass": no_halluc_count >= 2}

# ==================== 生成各任务函数（示例） ====================
def generate_summary(text: str, lang: str, theme: str) -> Dict:
    prompt = f"请总结以下文本，形成简洁摘要（{lang}）：" if lang == "zh" else "Please summarize the following text concisely:"
    # 调用 LLM 生成 answer...
    # 返回 {"question": prompt + text, "answer": summary}
    pass  # 实现细节请补充

# 类似实现阅读理解、多文本问答（≥3篇文本拼接）、对话补全（生成10+轮带[空白处]的对话）

# ==================== 主生成循环 ====================
task_counter = Counter()
lang_counter = Counter()
theme_counter = Counter()

sources = []  # 您需要提前准备：list of {"text": "...", "theme": "...", "source_url": "https://..."}

while len(data_list) < TARGET_TOTAL:
    task = random.choices(TASK_TYPES, weights=[1,1,1,1])[0]
    lang = "zh" if lang_counter["zh"] < TARGET_CN else "en"
    theme = random.choice(THEMES)
    
    # 根据 task 和 theme 选择/生成 context
    context = "..."  # 从 sources 选或合成
    
    if task == "多文本问答":
        # 拼接3+篇相关文本
        pass
    
    question, answer = generate_for_task(task, context, lang, theme)
    
    token_len = count_tokens(question + answer)
    if task == "多文本问答" and token_len < 8000:
        continue
    if task != "多文本问答" and not (4000 <= token_len <= 8000):
        if random.random() > 0.5: continue  # 控制比例
    
    # 幻觉检测
    detect = check_hallucination(context, question, answer)
    if not detect["pass"]:
        continue
    
    # 去重
    key = (question + answer).lower()
    if key in generated_set:
        continue
    m = get_minhash(key)
    if lsh.query(m):
        continue
    lsh.insert(str(uuid.uuid4()), m)
    generated_set.add(key)
    
    item = {
        "data_id": str(uuid.uuid4()),
        "task_type": task,
        "question": question,
        "source": sources[0]["source_url"],  # 实际替换
        "language": lang,
        "answer": answer,
        "content_theme": theme,
        "hallucination_detection_api_results": detect["results"]
    }
    data_list.append(item)
    
    task_counter[task] += 1
    lang_counter[lang] += 1
    theme_counter[theme] += 1
    
    if len(data_list) % 100 == 0:
        print(f"已生成 {len(data_list)} 条 | 任务分布: {task_counter}")

# 保存
with open("closed_domain_hallucination_data.jsonl", "w", encoding="utf-8") as f:
    for item in data_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("生成完成！")