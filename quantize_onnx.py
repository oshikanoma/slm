#!/usr/bin/env python3
"""Quantize the fp32 ONNX export to 4-bit (block-wise int4) for the browser.

Produces the layout Transformers.js expects:
    <out>/onnx/model_q4.onnx   (+ external data if any)
plus the tokenizer/config files copied alongside.

q4 (block int4) brings a 1.7B model from ~7.6GB fp32 to ~1-1.3GB.
"""
from __future__ import annotations
import os, shutil, sys

SRC = "onnx_build/qwen3-verifier-onnx"
OUT = "onnx_build/qwen3-verifier-tjs"
BLOCK = 32  # standard block size for browser int4


def main():
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig)
    import onnx

    os.makedirs(os.path.join(OUT, "onnx"), exist_ok=True)
    src_model = os.path.join(SRC, "model.onnx")
    print(f"[1/3] loading fp32 ONNX graph {src_model} ...")
    model = onnx.load(src_model, load_external_data=True)

    print(f"[2/3] 4-bit block quantize (block_size={BLOCK}, MatMulNBits) ...")
    # QOperator format -> MatMulNBits nodes, which is what ORT-web /
    # Transformers.js execute in the browser (QDQ int4 needs opset 21 and
    # isn't runnable by the web runtime).
    from onnxruntime.quantization.quant_utils import QuantFormat
    cfg = DefaultWeightOnlyQuantConfig(block_size=BLOCK, is_symmetric=True,
                                       quant_format=QuantFormat.QOperator)
    quant = MatMulNBitsQuantizer(model, algo_config=cfg)
    quant.process()

    out_model = os.path.join(OUT, "onnx", "model_q4.onnx")
    print(f"[3/3] saving -> {out_model}")
    # Keep weights inline if small enough; else external. Try inline first.
    onnx.save_model(quant.model.model, out_model,
                    save_as_external_data=True,
                    location="model_q4.onnx_data",
                    all_tensors_to_one_file=True)

    # Copy tokenizer + config (Transformers.js reads these from repo root).
    for f in ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json",
              "added_tokens.json", "merges.txt", "vocab.json", "chat_template.jinja"):
        s = os.path.join(SRC, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(OUT, f))
    print("done. files:")
    for root, _, files in os.walk(OUT):
        for fn in files:
            p = os.path.join(root, fn)
            print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
