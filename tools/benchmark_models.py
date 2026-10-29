"""Offline ASR benchmark for CSV rows: audio,reference_roman[,reference_native].

Usage:
  python tools/benchmark_models.py references.csv --models large-v3 medium

Use ``|`` between acceptable Roman spellings in ``reference_roman``. The CSV
must use human-checked reference transcripts and consented or licensed
audio. The tool does not record audio and refuses to fetch missing models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hinglish import romanize_hinglish
from src.settings import Settings


def words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        current = [i]
        for j, predicted in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != predicted)))
        previous = current
    return previous[-1] / len(ref)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def run(csv_path: Path, models: list[str]) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    settings = Settings()
    cache = settings.resolve_model_dir()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        if not row.get("audio") or not row.get("reference_roman"):
            raise ValueError("CSV needs audio and reference_roman columns")
        row["audio_path"] = str((csv_path.parent / row["audio"]).resolve())
        if not Path(row["audio_path"]).is_file():
            raise FileNotFoundError(row["audio_path"])

    results = {"dataset": str(csv_path), "samples": len(rows), "models": []}
    for name in models:
        repo = f"Systran/faster-whisper-{name}"
        try:
            model_dir = snapshot_download(repo, cache_dir=str(cache), local_files_only=True,
                                          allow_patterns=["config.json", "model.bin",
                                                          "tokenizer.json", "vocabulary.*",
                                                          "preprocessor_config.json"])
        except Exception as exc:
            raise RuntimeError(f"{name} is not cached; benchmark is offline") from exc
        init_start = time.perf_counter()
        model = WhisperModel(model_dir, device="cuda", compute_type="int8_float16")
        init_s = time.perf_counter() - init_start
        samples = []
        for row in rows:
            start = time.perf_counter()
            segments, info = model.transcribe(row["audio_path"], language="hi", beam_size=1)
            raw = " ".join(s.text.strip() for s in segments if s.text.strip())
            infer_s = time.perf_counter() - start
            roman = romanize_hinglish(raw)
            duration_s = float(info.duration or 0.0)
            samples.append({
                "audio": row["audio"],
                "reference_roman": row["reference_roman"],
                "reference_native": row.get("reference_native", ""),
                "raw": raw,
                "roman_hinglish": roman,
                "raw_wer": wer(row["reference_native"], raw)
                    if row.get("reference_native") else None,
                "roman_wer": min(
                    wer(candidate.strip(), roman)
                    for candidate in row["reference_roman"].split("|")
                    if candidate.strip()
                ),
                "roman_script_fraction": (
                    sum(ch.isascii() for ch in roman if ch.isalpha()) /
                    max(1, sum(ch.isalpha() for ch in roman))
                ),
                "inference_seconds": infer_s,
                "audio_seconds": duration_s,
                "real_time_factor": infer_s / duration_s if duration_s else None,
            })
        latencies = [s["inference_seconds"] for s in samples]
        results["models"].append({
            "model": name,
            "backend": "faster-whisper/CTranslate2 CUDA int8_float16",
            "initialization_seconds": init_s,
            "raw_wer_mean": statistics.mean(
                s["raw_wer"] for s in samples if s["raw_wer"] is not None
            ) if any(s["raw_wer"] is not None for s in samples) else None,
            "roman_wer_mean": statistics.mean(s["roman_wer"] for s in samples) if samples else None,
            "roman_script_fraction_mean": statistics.mean(
                s["roman_script_fraction"] for s in samples) if samples else None,
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "real_time_factor_mean": statistics.mean(
                s["real_time_factor"] for s in samples if s["real_time_factor"] is not None
            ) if samples else None,
            "samples": samples,
        })
        del model
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--models", nargs="+", default=["large-v3", "medium"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.csv, args.models)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
