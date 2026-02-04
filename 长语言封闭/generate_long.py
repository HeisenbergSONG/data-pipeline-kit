import json
import random
import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
import tiktoken
from datasketch import MinHash, MinHashLSH
from openai import OpenAI  # 支持 Qwen、GPT、Claude 等兼容接口
from transformers import AutoTokenizer  # Qwen tokenizer 精确长度

# ==================== 配置区 ====================
client = OpenAI(
    api_key="sk-你的密钥",          # Qwen: https://dashscope.aliyun.com
    base_url="https://dashscope.aliyun.com/compatible-mode/v1"  # Qwen推荐
)

MODEL = "qwen-max"                  # 或 qwen2.5-72b-instruct, gpt-4o 等
ENC = tiktoken.get_encoding("cl100k_base")   # 近似长度
QWEN_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  # 精确

# 来源文件夹（请提前准备）
SOURCES = {
    "legal": "sources/legal/",      # .txt 文件：民法典、判决书等
    "novel": "sources/novel/",      # 侦探/自传小说章节
    "code": "sources/code/",        # 多文件代码 + 注释
    "report": "sources/report/"     # 政府文件、学术、金融报告
}

DOMAINS = ["法律", "小说情节推理", "工程代码理解", "其他文本理解"]
TASK_TYPES = ["文本摘要", "阅读理解", "多文本问答", "对话补全"]
THEMES = ["新闻报道", "法律", "小说情节推理", "其他文本"]

# 幻觉检测模板（SOW原文）
HALLUCINATION_PROMPT = """Your task is to analyze a response to a user query and identify any instances of hallucinations based on the provided ground truth. ...
{context}
Response to Analyze: {response}
... 输出JSON格式（conflict和baseless_info均为0才通过）
"""

# ==================== 工具函数 ====================
def count_tokens(text: str, exact: bool = True) -> int:
    if exact:
        return len(QWEN_TOKENIZER.encode(text))
    return len(ENC.encode(text))

def llm_call(prompt: str, temperature=0.7) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2000
    )
    return resp.choices[0].message.content.strip()

def simulate_pass_rate(context: str, question: str, answer: str, n: int = 5) -> float:
    correct = 0
    for _ in range(n):
        pred = llm_call(f"Context: {context}\nQuestion: {question}\n请直接回答：")
        if any(k in pred.lower() for k in answer.lower().split()[:5]):  # 简单匹配
            correct += 1
    return round(correct / n, 2)

def run_hallucination_check(context: str, response: str) -> Dict:
    # 实际调用3个模型，这里用同一个模型跑3次模拟（生产环境换不同API）
    results = {}
    for i, m in enumerate(["gpt", "gemini", "claude"]):
        pred = llm_call(HALLUCINATION_PROMPT.format(context=context, response=response))
        try:
            res = json.loads(pred)
            results[m] = res["final_result"]
        except:
            results[m] = {"conflict": 1, "baseless_info": 1}
    return results

# MinHash LSH 去重
def build_lsh(samples: List[Dict], threshold: float = 0.85) -> MinHashLSH:
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, sample in enumerate(samples):
        key = f"{sample.get('question','')}{sample.get('answer','')}"
        m = MinHash()
        for d in key:
            m.update(d.encode('utf8'))
        lsh.insert(i, m)
    return lsh

def is_duplicate(sample: Dict, lsh: MinHashLSH, threshold: float = 0.85) -> bool:
    key = f"{sample.get('question','')}{sample.get('answer','')}"
    m = MinHash()
    for d in key:
        m.update(d.encode('utf8'))
    return len(lsh.query(m)) > 0

# ==================== 生成函数 ====================
def generate_long_sequence(language: str = "zh", target_tokens: int = 48000) -> Dict:
    domain = random.choice(DOMAINS)
    # 加载真实长文本（实际请放txt文件，这里用示例）
    # context = load_random_long_text(domain)  # 你需要实现
    context = "文档的基础信息如下：《全球通史》...（此处粘贴或读取你的长文本，实际需拼接多文件达target_tokens）" * (target_tokens // 500)

    prompt = f"""生成一条{language}的长序列数据，领域：{domain}。
要求：question基于context，answer唯一，推理跳数≥2，clues≥2条（positive优先）。
输出JSON：{{"question":..., "answer":..., "clues": {{"positive":[...],"negative":[...]}}, "reasoning_hops": int}}
Context: {context[:2000]}..."""   # 实际用完整context

    raw = json.loads(llm_call(prompt))
    pass_rate = simulate_pass_rate(context, raw["question"], raw["answer"])

    return {
        "id": f"long_{language}_{random.randint(1000,9999)}",
        "question": raw["question"],
        "context": context,
        "answer": raw["answer"],
        "language": language,
        "clues": raw["clues"],
        "domain": domain,
        "pass_rate": pass_rate,
        "reasoning_hops": raw.get("reasoning_hops", 3),
        "token_length": count_tokens(context, exact=True)
    }

def generate_hallucination_sample(task_type: str, language: str, theme: str) -> Dict:
    # 根据task_type加载对应上下文（多文本问答需3+篇）
    # 这里简化演示，实际请实现load_contexts(task_type)
    context = "示例上下文..." * 10
    prompt = f"生成{task_type}任务，语言{language}，主题{theme}。要求答案无幻觉、信息密集。"
    raw = json.loads(llm_call(prompt))

    detection = run_hallucination_check(context, raw["答案"])

    return {
        "data_id": f"hallu_{language}_{random.randint(1000,9999)}",
        "task_type": task_type,
        "题目": raw.get("题目", raw.get("question")),
        "来源": "https://example.com/改造新闻",
        "language": language,
        "答案": raw["答案"],
        "内容主题": theme,
        "hallucination_detection": detection,
        "token_length": count_tokens(raw.get("题目","") + context)
    }

# ==================== 主生成循环 ====================
def main(total: int = 100, ratio_long: float = 0.5):
    long_samples = []
    hallu_samples = []
    lsh_long = MinHashLSH(threshold=0.7, num_perm=128)
    lsh_hallu = MinHashLSH(threshold=0.85, num_perm=128)

    for i in tqdm(range(total)):
        if random.random() < ratio_long:
            sample = generate_long_sequence(
                language=random.choice(["zh", "en"]),
                target_tokens=random.randint(32000, 120000)
            )
            if not is_duplicate(sample, lsh_long, 0.7):
                long_samples.append(sample)
                # insert to lsh...
        else:
            sample = generate_hallucination_sample(
                task_type=random.choice(TASK_TYPES),
                language=random.choice(["zh", "en"]),
                theme=random.choice(THEMES)
            )
            if not is_duplicate(sample, lsh_hallu, 0.85):
                hallu_samples.append(sample)

    # 保存
    with open("long_sequence_samples.jsonl", "w", encoding="utf-8") as f:
        for s in long_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open("hallucination_samples.jsonl", "w", encoding="utf-8") as f:
        for s in hallu_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"生成完成：长序列 {len(long_samples)} 条，幻觉 {len(hallu_samples)} 条")

if __name__ == "__main__":
    main(total=200)   # 先跑小批量测试