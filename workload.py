import asyncio
import csv
import json
import random
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

SEED = 42
COUNT = 20
RATE = 0.5  # 每秒平均到达0.5个请求
rng = random.Random(SEED)

tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/models/Qwen3-0.6B",
    local_files_only=True
)

# 当前服务上下文限制4096，筛选可完整运行的记录，不截断长度。
eligible = []
with open("data/mooncake_trace.jsonl") as f:
    for line_number, line in enumerate(f, 1):
        if not line.strip():
            continue
        item = json.loads(line)
        n, m = int(item["input_length"]), int(item["output_length"])
        if n > 0 and m > 0 and n + m <= 4000:
            eligible.append({
                "source_line": line_number,
                "input_length": n,
                "output_length": m
            })

if len(eligible) < COUNT:
    raise RuntimeError(f"符合上下文限制的记录仅{len(eligible)}条")

sample = rng.sample(eligible, COUNT)
jobs = []
arrival = 0.0
for i, item in enumerate(sample):
    # 指数分布的到达间隔对应Poisson到达过程。
    arrival += rng.expovariate(RATE)
    n = item["input_length"]
    prompt = " hello" * n
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual != n:
        raise RuntimeError(f"构造输入长度不一致: target={n}, actual={actual}")
    jobs.append({
        **item, "request_id": i + 1,
        "arrival_s": arrival, "prompt": prompt
    })

Path("results").mkdir(exist_ok=True)
Path("results/sampled_workload.json").write_text(
    json.dumps({
        "seed": SEED, "rate_requests_per_second": RATE,
        "selection_rule": "input_length > 0, output_length > 0, sum <= 4000",
        "requests": jobs
    }, ensure_ascii=False, indent=2)
)

async def send(session, job, start):
    await asyncio.sleep(max(0, start + job["arrival_s"] - time.perf_counter()))
    sent = time.perf_counter()
    row = {
        "request_id": job["request_id"],
        "source_line": job["source_line"],
        "scheduled_arrival_s": round(job["arrival_s"], 4),
        "actual_send_s": round(sent - start, 4),
        "target_input_tokens": job["input_length"],
        "target_output_tokens": job["output_length"],
        "input_tokens": "",
        "output_tokens": "",
        "status": "",
        "latency_s": "",
        "error": ""
    }
    try:
        async with session.post(
            "http://127.0.0.1:30000/v1/completions",
            json={
                "model": "Qwen/Qwen3-0.6B",
                "prompt": job["prompt"],
                "max_tokens": job["output_length"],
                "temperature": 0,
                "ignore_eos": True,
                "stream": False
            }
        ) as response:
            body = await response.text()
            row["latency_s"] = round(time.perf_counter() - sent, 4)
            row["status"] = response.status
            if response.status == 200:
                data = json.loads(body)
                row["input_tokens"] = data["usage"]["prompt_tokens"]
                row["output_tokens"] = data["usage"]["completion_tokens"]
            else:
                row["error"] = body[:500]
    except Exception as exc:
        row["status"] = "ERROR"
        row["latency_s"] = round(time.perf_counter() - sent, 4)
        row["error"] = str(exc)
    print(
        f"request={row['request_id']:02d} "
        f"input={row['input_tokens']} output={row['output_tokens']} "
        f"status={row['status']} latency={row['latency_s']}s",
        flush=True
    )
    return row

async def main():
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()
        rows = await asyncio.gather(*(send(session, j, start) for j in jobs))
    with open("results/workload_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successful = [r for r in rows if r["status"] == 200]
    print(f"\n完成：成功 {len(successful)}/{len(rows)}")
    if successful:
        mean = sum(r["latency_s"] for r in successful) / len(successful)
        print(f"平均总延迟：{mean:.4f}s")
    print("结果文件：results/workload_results.csv")

asyncio.run(main())
