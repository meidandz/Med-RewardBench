#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
CUDA_VISIBLE_DEVICES=1 python vlm/00_benchmark_full_pipeline.py \
  --models  MedMO-8B \
  --modalities abdomen brain breast chest eye foot gastrointestinal_tract heart lower_limb lung oral_cavity pelvic_cavity upper_limb \
  --data_root /data/dingmeidan/LLM/MedPrefer-Bench/data/ \
  --excel_output ./output/MedMO-8B.xlsx

QWEN_ENABLE_THINKING=0 CUDA_VISIBLE_DEVICES=0 python vlm/00_benchmark_full_pipeline.py \
  --models MedMO-8B \
  --modalities abdomen brain breast chest eye foot gastrointestinal_tract heart lower_limb lung oral_cavity pelvic_cavity upper_limb \
  --data_root /data/dingmeidan/LLM/MedPrefer-Bench/data/ \
  --excel_output ./output/MedMO-8B-nothinking.xlsx \
  --extra_output_formats csv jsonl \
  --table_output_dir ./output/tables \
  --prompt_thinking_mode auto



'''

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
import json
import time
import re
import io
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from prompt import get_prompt_ours
from model_builder_chosen_respon import get_chosen_response
from PIL import Image


def is_thinking_prompt_enabled(args):
    mode = args.prompt_thinking_mode
    if mode == "thinking":
        return True
    if mode == "nothinking":
        return False
    # auto mode follows Qwen thinking env switch
    return os.getenv("QWEN_ENABLE_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}


def retry(attempts=3, delay=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1}/{attempts} failed: {e}")
                    if i < attempts - 1:
                        time.sleep(delay)
            return None
        return wrapper
    return decorator


def construct_input(prompt_dict, judge_mode, setting, instruction, responses):
    prompt = prompt_dict + "\nHere is the input:\n"
    if judge_mode == "score":
        prompt += f"[The Start of User Instruction]\n{instruction}\n[The End of User Instruction]\n[The Start of Assistant's Answer]\n{responses[0]}\n[The End of Assistant's Answer]"
    elif judge_mode == 'pair':
        prompt += f"[The Start of User Instruction]\n{instruction}\n[The End of User Instruction]\n[The Start of Assistant A's Answer]\n{responses[0]}\n[The End of Assistant A's Answer]\n[The Start of Assistant B's Answer]\n{responses[1]}\n[The End of Assistant B's Answer]"
    elif judge_mode == 'batch':
        assistant_name = "A"
        prompt += f"[The Start of User Instruction]\n{instruction}\n[The End of User Instruction]\n"
        for i, response in enumerate(responses):
            prompt += f"[The Start of Assistant {assistant_name}'s Answer]\n{response}\n[The End of Assistant {assistant_name}'s Answer]\n"
            assistant_name = chr(ord(assistant_name) + 1)
    return prompt


def benchmark(args):

    parquet_path = os.path.join(args.data_root,"pair_gt_all/parquet/", f"{args.modality_type}.parquet")
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    output_path = f"./data/benchmark_result_parquet1/{args.modality_type}/{args.judge_mode}_{args.model}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prompt_dict = get_prompt_ours(thinking_enabled=is_thinking_prompt_enabled(args))
    multi_modal_inputs = []

    for item in tqdm(dataset, desc=f"Evaluating {args.modality_type}"):
        image = item['image']
        responses = [item['answer1_text'], item['answer2_text']]
        prompt = construct_input(prompt_dict, args.judge_mode, args.setting, item['question'], responses)
        multi_modal_inputs.append({
            "image": image,
            "prompt": prompt,
            "metadata": item
        })

    get_chosen_response(args, args.model, multi_modal_inputs, output_path)
    return output_path

def benchmark_all_modalities(args, model_name, modalities):
    prompt_dict = get_prompt_ours(thinking_enabled=is_thinking_prompt_enabled(args))
    all_inputs = []

    for modality in modalities:
        print(f"加载模态数 {modality}")
        parquet_path = os.path.join(args.data_root, "pair_gt_all/parquet/", f"{modality}.parquet")
        dataset = load_dataset("parquet", data_files=parquet_path, split="train")

        for item in dataset:
            image = item['image']
            responses = [item['answer1_text'], item['answer2_text']]
            prompt = construct_input(prompt_dict, args.judge_mode, args.setting, item['question'], responses)

            all_inputs.append({
                "image": image,
                "prompt": prompt,
                "metadata": item,
                "modality": modality            })

    print(f"{len(all_inputs)} 条输入，准备推理")
    
    output_path = f"./data/benchmark_result_parquet1/all/{args.judge_mode}_{model_name}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    get_chosen_response(args, model_name, all_inputs, output_path)

    return output_path  # 返回总文件路�?


def normalize_chosen_answer(answer_str):
    answer_str = answer_str.strip().strip('"')
    dimensions = ['accuracy', 'relevance', 'comprehensiveness', 'creativity', 'responsiveness', 'overall']

    # 1) Prefer parsing JSON output (strict prompt target).
    json_candidate = answer_str
    if "{" in answer_str and "}" in answer_str:
        json_candidate = answer_str[answer_str.find("{"):answer_str.rfind("}") + 1]
    try:
        parsed_json = json.loads(json_candidate)
        if isinstance(parsed_json, dict):
            normalized = {}
            for dim in dimensions:
                val = str(parsed_json.get(dim, "")).strip().upper()
                if val not in {"A", "B"}:
                    normalized = None
                    break
                normalized[dim] = val
            if normalized is not None:
                if "reason" in parsed_json:
                    normalized["reason"] = str(parsed_json.get("reason", "")).strip()
                return normalized
    except Exception:
        pass

    # 2) Parse structured text by dimension.
    parsed = {}
    for dim in dimensions:
        m = re.search(rf'"?{dim}"?\s*:\s*"?([AB])"?', answer_str, flags=re.IGNORECASE)
        if m:
            parsed[dim] = m.group(1).upper()
    if len(parsed) == len(dimensions):
        reason_match = re.search(r'"?reason"?\s*:\s*"([^"]*)"', answer_str, flags=re.IGNORECASE)
        if reason_match:
            parsed["reason"] = reason_match.group(1).strip()
        return parsed

    # 3) If model only outputs one choice, default all dimensions to that choice.
    single_match = re.search(r'\[\[([AB])\]\]', answer_str, flags=re.IGNORECASE)
    if single_match:
        choice = single_match.group(1).upper()
        return {dim: choice for dim in dimensions}

    standalone_choices = re.findall(r'\b([AB])\b', answer_str, flags=re.IGNORECASE)
    if len(standalone_choices) == 1:
        choice = standalone_choices[0].upper()
        return {dim: choice for dim in dimensions}

    if answer_str.upper() in {"A", "B"}:
        return {dim: answer_str.upper() for dim in dimensions}

    # 无法解析为有效结果
    return None



def process_jsonl_file(input_file, output_file):
    if os.path.abspath(input_file) == os.path.abspath(output_file):
        raise ValueError(f"input_file and output_file must be different: {input_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    skipped_count = 0
    total = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            total += 1
            try:
                data = json.loads(line.strip())
                if 'chosen_answer' in data:
                    normalized_answer = normalize_chosen_answer(data['chosen_answer'])
                    
                    if normalized_answer is None:
                        skipped_count += 1
                        continue  # �?跳过不合法答�?    
                    data['chosen_answer'] = normalized_answer                
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error processing line: {e}")
                skipped_count += 1
    
    print(f"标准化完成：共处理{total} 条，跳过无效 {skipped_count}")




def evaluate_accuracy(gt_path, pred_path, modality, model):
    with open(gt_path, 'r', encoding='utf-8') as f1, open(pred_path, 'r', encoding='utf-8') as f2:
        gt_data = {json.loads(l)['question_id']: json.loads(l)['human_answer'] for l in f1}
        pred_data = {json.loads(l)['question_id']: json.loads(l)['chosen_answer'] for l in f2}

    dimensions = ['accuracy', 'relevance', 'comprehensiveness', 'creativity', 'responsiveness', 'overall']
    metrics = {dim: {'total': 0, 'correct': 0} for dim in dimensions}

    for qid in gt_data:
        for dim in dimensions:
            metrics[dim]['total'] += 1
            pred_answer = pred_data.get(qid, {})
            if isinstance(pred_answer, dict) and pred_answer.get(dim) == gt_data[qid][dim]:
                metrics[dim]['correct'] += 1

    result_row = {
        "modality": modality,
        "model": model
    }
    for dim in dimensions:
        total = metrics[dim]['total']
        correct = metrics[dim]['correct']
        acc = correct / total if total > 0 else 0
        result_row[f"{dim}_acc"] = round(acc, 4)
    return result_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', default=["Qwen2-VL-7B-sft", "Qwen2-VL-7B-dpo", "MBZUAI/MedMO-8B"])
    parser.add_argument("--modalities", nargs='+', default=["abdomen", "brain", "chest"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--judge_mode", type=str, default='pair')
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--setting", type=str, default="No COT")
    parser.add_argument("--excel_output", type=str, default="evaluation_summary.xlsx")
    parser.add_argument(
        "--prompt_thinking_mode",
        type=str,
        default="auto",
        choices=["auto", "thinking", "nothinking"],
        help="Prompt template mode. auto follows QWEN_ENABLE_THINKING env."
    )
    parser.add_argument(
        "--extra_output_formats",
        nargs='*',
        default=["csv", "jsonl"],
        choices=["csv", "jsonl"],
        help="Additional output formats besides xlsx."
    )
    parser.add_argument(
        "--table_output_dir",
        type=str,
        default=None,
        help="Directory for csv/jsonl outputs. Defaults to excel_output directory."
    )
    args = parser.parse_args()
    table_output_dir = args.table_output_dir or os.path.dirname(args.excel_output) or "."
    os.makedirs(table_output_dir, exist_ok=True)
    all_results = []
    excel_stem = Path(args.excel_output).stem
    single_model_mode = len(args.models) == 1

    with pd.ExcelWriter(args.excel_output, engine='xlsxwriter') as writer:
        for model in args.models:
            print(f"\n当前模型: {model}")
            args.model = model  # 填补 args.model �?get_chosen_response �?            
            model_results = []

            # 🧠 只推理一次，输出为一�?jsonl 文件
            combined_result_path = benchmark_all_modalities(args, model, args.modalities)

            # 🔁 分类保存每个模态的 jsonl，评估准确率
            with open(combined_result_path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line.strip()) for line in f]

            by_modality = {}
            for entry in lines:
                # print(entry)
                # quit()
                modality = entry['modality_type']
                by_modality.setdefault(modality, []).append(entry)

            for modality, entries in by_modality.items():
                print(f"模态 {modality} - 保存输出 & 准确率评")

                raw_path = f"./data/benchmark_result_parquet1/{modality}/{args.judge_mode}_{model}.jsonl"
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                with open(raw_path, 'w', encoding='utf-8') as f_out:
                    for entry in entries:
                        f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

                norm_path = f"./data/benchmark_result_parquet_normalized1/{modality}/{args.judge_mode}_{model}.jsonl"
                process_jsonl_file(raw_path, norm_path)

                gt_path = os.path.join(args.data_root, "pair_gt_all", f"{modality}.jsonl")
                result_row = evaluate_accuracy(gt_path, norm_path, modality, model)
                model_results.append(result_row)

            # 📄 写入 Excel
            df = pd.DataFrame(model_results)
            df.to_excel(writer, sheet_name=model[:31], index=False)
            all_results.extend(model_results)

            safe_model_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', model)
            if single_model_mode:
                table_base_name = excel_stem
            else:
                table_base_name = f"{excel_stem}__{safe_model_name}"
            if "csv" in args.extra_output_formats:
                csv_path = os.path.join(table_output_dir, f"{table_base_name}.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            if "jsonl" in args.extra_output_formats:
                jsonl_path = os.path.join(table_output_dir, f"{table_base_name}.jsonl")
                df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)

    all_df = pd.DataFrame(all_results)
    if len(all_df) > 0:
        if "csv" in args.extra_output_formats:
            all_csv_path = os.path.join(table_output_dir, f"{excel_stem}.csv")
            all_df.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
        if "jsonl" in args.extra_output_formats:
            all_jsonl_path = os.path.join(table_output_dir, f"{excel_stem}.jsonl")
            all_df.to_json(all_jsonl_path, orient='records', lines=True, force_ascii=False)

    print(f"\n所有模型和模态的评估结果已保存至：{args.excel_output}")


if __name__ == '__main__':
    main()
