#!/usr/bin/env python
"""Дообучение текстового энкодера на парах запрос–объявление."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG, load_config, resolve_project_path
from src.data import load_train
from src.text import normalize_text
from src.validation import split_by_query_signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-dir", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--output-dir", default="artifacts/models/avito-minilm-retriever")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=96)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    train = load_train(resolve_project_path(args.data_dir))
    train_part, _ = split_by_query_signature(
        train, float(config["validation_size"]), int(config["random_seed"])
    )
    work = train_part.copy()
    work["_query"] = [
        " ".join(filter(None, [normalize_text(query), normalize_text(params)]))
        for query, params in zip(work["search_query"], work["search_infn_params_text"])
    ]
    work["_item"] = [
        " ".join(filter(None, [normalize_text(title), normalize_text(params)]))
        for title, params in zip(work["item_title_raw"], work["item_infn_params_text"])
    ]
    work = work[(work["_query"] != "") & (work["_item"] != "")]
    work = work.sample(frac=1.0, random_state=int(config["random_seed"])).drop_duplicates("_query")
    examples = [InputExample(texts=[query, item]) for query, item in zip(work["_query"], work["_item"])]
    seed = int(config["random_seed"])
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        examples, shuffle=True, batch_size=args.batch_size, drop_last=True,
        num_workers=0, generator=generator,
    )

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model_source = Path(args.model_dir)
    model_name = str(resolve_project_path(args.model_dir)) if model_source.exists() else args.model_dir
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = args.max_seq_length
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = len(loader) * args.epochs
    output = resolve_project_path(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Training pairs={len(examples)}, batch={args.batch_size}, epochs={args.epochs}, "
        f"steps={steps}, warmup={math.ceil(steps * 0.1)}"
    )
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=math.ceil(steps * 0.1),
        optimizer_params={"lr": 2e-5},
        output_path=str(output),
        use_amp=device.startswith("cuda"),
        show_progress_bar=True,
        checkpoint_path=str(output.parent / "avito-minilm-checkpoints"),
        checkpoint_save_steps=max(250, len(loader)),
        checkpoint_save_total_limit=2,
    )
    print(f"Saved fine-tuned model to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
