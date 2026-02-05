import json
import random
import os
from collections import Counter
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer
from openai import OpenAI
from typing import Tuple

# ==================== 配置 ====================
client = OpenAI(
    api_key="sk-",  # ← 修改此处
    base_url="https://dashscope.aliyun.com/compatible-mode/v1"
)
MODEL = "qwen-long"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

DOMAINS = ["法律", "小说情节推理", "工程代码理解", "其他文本理解"]
LANGUAGES = ["zh", "en"]

SOURCES_DIR = {
    "法律": "sources/legal",
    "小说情节推理": "sources/novel",
    "工程代码理解": "sources/code",
    "其他文本理解": "sources/report"
}

# ==================== 完整版 load_long_context ====================
def load_long_context(domain: str, target_tokens: int, min_tokens: int = 32000, max_tokens: int = 120000) -> Tuple[str, int]:
    folder = SOURCES_DIR.get(domain)
    if not folder or not os.path.exists(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    txt_files = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"文件夹 {folder} 中没有 .txt 文件")

    random.shuffle(txt_files)
    selected_files = []
    context_parts = []
    current_tokens = 0

    # 策略1: 优先单个完美匹配文件
    for file in txt_files[:8]:
        filepath = os.path.join(folder, file)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        tokens = len(tokenizer.encode(text))
        if min_tokens <= tokens <= max_tokens:
            return text, tokens
        elif tokens > max_tokens:
            encoded = tokenizer.encode(text)
            truncated = tokenizer.decode(encoded[:max_tokens])
            return truncated, max_tokens
        elif tokens >= 10000:
            selected_files.append(file)
            context_parts.append(text)
            current_tokens = tokens
            break

    # 策略2: 拼接多个文件
    if current_tokens < min_tokens:
        for file in txt_files:
            if file in selected_files:
                continue
            filepath = os.path.join(folder, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            tokens = len(tokenizer.encode(text))
            if current_tokens + tokens > max_tokens:
                break
            selected_files.append(file)
            context_parts.append(f"\n\n=== 文档片段 [{file}] ===\n\n{text}")
            current_tokens += tokens
            if current_tokens >= target_tokens:
                break

    context = "".join(context_parts)
    actual_tokens = len(tokenizer.encode(context))

    if actual_tokens > max_tokens:
        encoded = tokenizer.encode(context)
        context = tokenizer.decode(encoded[:max_tokens])
        actual_tokens = max_tokens
    elif actual_tokens < min_tokens:
        context += "\n\n[补充说明：以上内容由多份相关文档综合整理而成]"

    return context, actual_tokens

# ==================== 辅助函数 ====================
def llm_call(prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""

def simulate_pass_rate(context: str, question: str, answer: str, n: int = 5) -> float:
    correct = 0
    for _ in range(n):
        pred = llm_call(f"Context: {context[:10000]}...\nQuestion: {question}\n请仅基于context回答：", temperature=0.0)
        if sum(1 for kw in answer.split()[:6] if kw.lower() in pred.lower()) >= 3:
            correct += 1
    return round(correct / n, 2)

def simulate_clue_validation(context: str, question: str, clues: dict, trials: int = 2) -> dict:
    positive_clues = clues.get("positive", [])
    clue_str = "\n".join(f"线索{i+1}: {c}" for i, c in enumerate(positive_clues))
    prompt = f"""仅使用以下线索回答问题，不得使用额外知识：
{clue_str}

问题：{question}

请给出完整答案："""
    passes = 0
    for _ in range(trials):
        pred = llm_call(prompt, temperature=0.0)
        if any(kw in pred for kw in ["正确", "符合", "根据线索", "答案是"]):
            passes += 1
    return {"passes": passes, "trials": trials, "pass_rate": round(passes / trials, 2), "valid": passes >= 1}

def is_duplicate(sample: dict, lsh: MinHashLSH) -> bool:
    key = f"{sample['question']}{sample['answer']}{json.dumps(sample.get('clues', {}))}"
    m = MinHash(num_perm=128)
    for d in key.lower():
        m.update(d.encode('utf8'))
    return len(lsh.query(m)) > 0

# ==================== 生成单条数据 ====================
def generate_one_sample(domain: str, language: str, target_tokens: int) -> dict | None:
    try:
        context, actual_tokens = load_long_context(domain, target_tokens)
    except Exception as e:
        print(f"加载context失败 ({domain}): {e}")
        return None

    if not (32000 <= actual_tokens <= 120000):
        return None

    prompt = f"""生成一条{language}的长序列训练数据，领域：{domain}。
要求：question基于context闭环，answer唯一，推理跳数>=2，clues至少2条（positive优先）。
输出严格JSON格式：
{{"question": "...", "answer": "...", "clues": {{"positive": [...], "negative": [...]}}, "reasoning_hops": int}}"""

    try:
        raw = json.loads(llm_call(prompt + f"\nContext preview: {context[:4000]}..."))
    except:
        return None

    pass_rate = simulate_pass_rate(context, raw["question"], raw["answer"], n=5)
    clue_val = simulate_clue_validation(context, raw["question"], raw["clues"])

    return {
        "id": f"long_{language}_{domain[:2]}_{random.randint(10000,99999)}",
        "question": raw["question"],
        "context": context,
        "answer": raw["answer"],
        "language": language,
        "clues": raw["clues"],
        "domain": domain,
        "pass_rate": pass_rate,
        "reasoning_hops": raw.get("reasoning_hops", 3),
        "token_length": actual_tokens,
        "clue_validation": clue_val
    }

# ==================== 主函数（完整版） ====================
def main(total: int = 50000):
    samples = []
    lsh = MinHashLSH(threshold=0.7, num_perm=128)
    domain_counter = Counter()
    length_distribution = {"32k_48k": 0, "48k_plus": 0}

    target_48k_ratio = 0.60

    pbar = tqdm(total=total, desc="生成长序列数据")
    attempts = 0
    max_attempts = total * 3  # 防止死循环

    while len(samples) < total and attempts < max_attempts:
        attempts += 1
        domain = random.choice(DOMAINS)
        if domain_counter[domain] >= total // 4 + 2000:
            continue

        language = random.choice(LANGUAGES)

        # 控制48k+比例
        current_48k_ratio = length_distribution["48k_plus"] / (len(samples) + 1) if samples else 0
        if current_48k_ratio < target_48k_ratio:
            target_tokens = random.randint(48000, 120000)
        else:
            target_tokens = random.randint(32000, 47999)

        sample = generate_one_sample(domain, language, target_tokens)
        if not sample:
            continue

        if is_duplicate(sample, lsh):
            continue

        # 插入LSH
        key = f"{sample['question']}{sample['answer']}{json.dumps(sample.get('clues', {}))}"
        m = MinHash(num_perm=128)
        for d in key.lower():
            m.update(d.encode('utf8'))
        lsh.insert(len(samples), m)

        samples.append(sample)
        domain_counter[domain] += 1

        tokens = sample["token_length"]
        if tokens >= 48000:
            length_distribution["48k_plus"] += 1
        else:
            length_distribution["32k_48k"] += 1

        pbar.update(1)

    pbar.close()

    # 保存
    with open("long_sequence_data.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ 生成完成！总条数: {len(samples)}")
    print("领域分布:", dict(domain_counter))
    print("语言分布:", Counter(s["language"] for s in samples))
    print(f"48k+ 比例: {length_distribution['48k_plus'] / len(samples):.3f}")
    print("文件已保存: long_sequence_data.jsonl")

if __name__ == "__main__":
    main(total=200)   # 先测试200条，确认无误后再改为50000