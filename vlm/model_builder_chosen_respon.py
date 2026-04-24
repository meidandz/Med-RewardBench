"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.

"""

# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 python model_builder.py
from transformers import AutoTokenizer, AutoProcessor

from vllm import LLM, SamplingParams
# from vllm.assets.image import ImageAsset
# from vllm.utils import FlexibleArgumentParser
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
import json
from tqdm import tqdm
from PIL import Image

# Input image and question
# image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
# question = "What is the content of this image?"


def _build_chat_template_handler(model_name: str):
    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _enable_qwen_thinking() -> bool:
    return os.getenv("QWEN_ENABLE_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _infer_tensor_parallel_size(preferred: int = 2) -> int:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        # e.g. "0", "0,1", "0,1,2"
        num_visible = len([x for x in visible.split(",") if x.strip() != ""])
        return max(1, min(preferred, num_visible))
    return preferred


def _require_env_model_path(env_var: str, example_value: str) -> str:
    model_path = os.getenv(env_var, "").strip()
    if model_path:
        return model_path
    raise ValueError(
        f"Model path for this checkpoint is not configured. "
        f"Set environment variable {env_var} to the local checkpoint path "
        f"(for example: {example_value})."
    )


# LLaVA-1.5-7B: llava-hf/llava-1.5-7b-hf
def run_llava1_5_7_build():
    llm = LLM(model="llava-hf/llava-1.5-7b-hf",
            #   tensor_parallel_size=2,
              gpu_memory_utilization=0.9)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def run_stllavamed_build():
    llm = LLM(model="ZachSun/stllava-med-7b",
            #   tensor_parallel_size=2,
              gpu_memory_utilization=0.3)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_llava1_5_13_build():
    llm = LLM(model="llava-hf/llava-1.5-13b-hf",
              tensor_parallel_size=2,
              gpu_memory_utilization=0.9)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def llava1_5_inference(question, tokenizer):

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    return prompt


# LLaVA-1.6-7B: llava-hf/llava-v1.6-mistral-7b-hf
def run_llava1_6_7_build():

    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", gpu_memory_utilization=0.5)
    tokenizer = None
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_llava1_6_13_build():

    llm = LLM(model="llava-hf/llava-v1.6-vicuna-13b-hf",tensor_parallel_size=2,gpu_memory_utilization=0.9)
    tokenizer = None
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_llava1_6_34_build():

    llm = LLM(model="llava-hf/llava-v1.6-34b-hf",tensor_parallel_size=2, gpu_memory_utilization=0.9,max_model_len=4096,)
    tokenizer = None
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids
def llava1_6_inference(question, tokenizer):

    prompt = f"[INST] <image>\n{question} [/INST]"

    return prompt

# LLaVA-OneVision: llava-hf/llava-onevision-qwen2-7b-ov-hf
def run_llava_onevision7_build():

    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
              max_model_len=16384,tensor_parallel_size=2,gpu_memory_utilization=0.9)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def run_llava_onevision72_build():

    llm = LLM(model="llava-hf/llava-onevision-qwen2-72b-ov-hf",
            #   max_model_len=4096,
              tensor_parallel_size=2,
              enforce_eager=True,
              gpu_memory_utilization=0.9)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def llava_onevision_inference(question,tokenizer):
    
    prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"
    return prompt


# Fuyu-8B: adept/fuyu-8b
def run_fuyu_build():

    llm = LLM(model="adept/fuyu-8b")
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def fuyu_inference(question, tokenizer):

    prompt = f"{question}\n"
    
    return prompt


# Phi-3-Vision: microsoft/Phi-3-vision-128k-instruct
def run_phi3v_build():

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model="microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True,
        max_num_seqs=5,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def phi_inference(question, tokenizer):

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501

    return prompt


# Phi-3-5-Vision: microsoft/Phi-3.5-vision-instruct
def run_phi3_5v_build():

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_num_seqs=5,
        mm_processor_kwargs={"num_crops": 16},
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=2,
    )
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids


# PaliGemma-3B: google/paligemma-3b-mix-224
def run_paligemma_build():

    # PaliGemma has special prompt format for VQA

    llm = LLM(model="google/paligemma-3b-mix-224",gpu_memory_utilization=0.6,)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def paligemma_inference(question, tokenizer):

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"

    return prompt


# Chameleon-7B: facebook/chameleon-7b
def run_chameleon_build():

    llm = LLM(model="facebook/chameleon-7b",tensor_parallel_size=2,gpu_memory_utilization=0.9,)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def chameleon_inference(question, tokenizer):

    prompt = f"{question}<image>"

    return prompt


# MiniCPM-V2-5: openbmb/MiniCPM-Llama3-V-2_5
def run_minicpmv2_5_build():

    # 2.5
    model_name = "openbmb/MiniCPM-Llama3-V-2_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
    )

    stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    return llm, tokenizer, stop_token_ids
def minicpmv_inference(question, tokenizer):

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt

# MiniCPM-V2-6: openbmb/MiniCPM-V-2_6
def run_minicpmv2_6_build():


    model_name = "openbmb/MiniCPM-V-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )

    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, tokenizer, stop_token_ids


# InternVL2-8B: OpenGVLab/InternVL2-8B
def run_internvl8b_build():
    # model_name = "OpenGVLab/InternVL2-8B"
    # model_name = "OpenGVLab/InternVL2_5-8B"
    model_name = "OpenGVLab/InternVL3-8B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        # tensor_parallel_size=2,
        gpu_memory_utilization=0.25,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, tokenizer, stop_token_ids
def run_internvl40b_build():
    model_name = "OpenGVLab/InternVL2-40B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, tokenizer, stop_token_ids
def run_internvl15_build():
    model_name = "OpenGVLab/InternVL-Chat-V1-5"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, tokenizer, stop_token_ids

def internvl_inference(question, tokenizer):

    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return prompt


# BLIP-2: Salesforce/blip2-opt-2.7b
def run_blip2_build():

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa

    llm = LLM(model="Salesforce/blip2-opt-2.7b",tensor_parallel_size=2,gpu_memory_utilization=0.4,)
        #tensor_parallel_size=2,)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def blip2_inference(question, tokenizer):

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"

    return prompt


# Qwen2-VL: Qwen/Qwen2-VL-7B-Instruct
def run_qwen2vl7_build():
    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
        gpu_memory_utilization=0.9,
        #tensor_parallel_size=2,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_qwen3vl32_build():
    model_name = "Qwen/Qwen3-VL-32B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
        gpu_memory_utilization=0.98,
        tensor_parallel_size=2,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_medvlmr1_build():
    model_name = "JZPeterPan/MedVLM-R1"

    # Tested on L40
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=2,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_medr1_build():
    model_name = _require_env_model_path("MEDR1_MODEL_PATH", "/path/to/Med-R1/VQA_MRI")

    # Tested on L40
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=2,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_qwen2vl7_sft_build():
    model_name = _require_env_model_path("QWEN2_VL_7B_SFT_PATH", "/path/to/qwen2_vl_lora_sft")

    # Tested on L40
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=2,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids


def run_qwen2_5vl7_sft_build():
    model_name = _require_env_model_path("QWEN2_5_VL_7B_SFT_PATH", "/path/to/qwen2.5-vl-7b-sft")

    # Tested on L40
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
        gpu_memory_utilization=0.95,
        # tensor_parallel_size=2,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_qwen2vl7_dpo_build():
    model_name = _require_env_model_path("QWEN2_VL_7B_DPO_PATH", "/path/to/qwen2_vl_lora_dpo")

    # Tested on L40
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        trust_remote_code=True,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #     "min_pixels": 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        # },
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=2,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids
def run_qwen2vl2_build():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        # max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={
        #    "min_pixels": 28 * 28,
        #    "max_pixels": 1280 * 28 * 28,
        #},
        gpu_memory_utilization=0.5,
        tensor_parallel_size=2,
        # disable_custom_all_reduce=True,
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids
def run_qwen2vl72_build():
    model_name = "Qwen/Qwen2-VL-72B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids
def qwen2vl_inference(question, tokenizer):
    enable_thinking = _enable_qwen_thinking()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    # try:
    #     return tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #         enable_thinking=enable_thinking,
    #     )
    # except TypeError:
    #     return tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    # except Exception:
    #     # Fallback for tokenizers/processors that do not support multimodal messages.
    #     prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    #               "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    #               f"{question}<|im_end|>\n"
    #               "<|im_start|>assistant\n")
    #     return prompt

# Qwen2.5-VL
def run_qwen2_5_vl7_build():

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.3,
        # tensor_parallel_size=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        # limit_mm_per_prompt={modality: 1},
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_r1reward_build():

    model_name = "yifanzhang114/R1-Reward"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        # limit_mm_per_prompt={modality: 1},
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_lingshu7_build():

    model_name = "lingshu-medical-mllm/Lingshu-7B"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        # limit_mm_per_prompt={modality: 1},
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_medmo8_build():

    model_name = "MBZUAI/MedMO-8B"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.25,
        # tensor_parallel_size=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_medix_r1_8b_build():

    model_name = "MBZUAI/MediX-R1-8B"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.3,
        # tensor_parallel_size=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_qwen3_vl32_build():

    model_name = "Qwen/Qwen3-VL-32B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=_infer_tensor_parallel_size(2),
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        # limit_mm_per_prompt={modality: 1},
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_qwen3_vl8_build():

    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.25,
        tensor_parallel_size=_infer_tensor_parallel_size(2),
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        # limit_mm_per_prompt={modality: 1},
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_qwen2_5_vl72_build():

    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.98,
        tensor_parallel_size=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            # "fps": 1,
        },
        # limit_mm_per_prompt={modality: 1},
    )

    tokenizer = _build_chat_template_handler(model_name)
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids
    
def run_llavacritic_build():

    model_name = "lmms-lab/LLaVA-Critic-R1-7B-Plus-Qwen"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        gpu_memory_utilization=0.4,
        # tensor_parallel_size=2,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            # "fps": 1,
        },
        # limit_mm_per_prompt={modality: 1},
    )

    tokenizer =None
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

# Pixtral-12B: mistral-community/pixtral-12b
def run_pixtral_hf_build():

    model_name = "mistral-community/pixtral-12b"

    llm = LLM(
        model=model_name,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
    )
    tokenizer =None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def pixtral_hf_inference(question, tokenizer):
    prompt = f"<s>[INST]{question}\n[IMG][/INST]"
    return prompt
# <s>[SYSTEM_PROMPT] <system prompt>[/SYSTEM_PROMPT][INST] <user message>[/INST] <assistant response></s>[INST] <user message>[/INST]



def run_huatuo_build():

    llm = LLM(model="FreedomIntelligence/HuatuoGPT-Vision-7B-hf",
              max_model_len=16384,gpu_memory_utilization=0.9,tensor_parallel_size=2,trust_remote_code=True,)
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids



# LLama 3.2-11B: meta-llama/Llama-3.2-11B-Vision-Instruct
def run_mllama11_build():

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
    )
    tokenizer =None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def run_mllama90_build():

    model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
    )
    tokenizer =None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def mllama_inference(question, tokenizer):

    prompt = f"<|image|><|begin_of_text|>{question}"

    return prompt


# Molmo-7B: allenai/Molmo-7B-D-0924
def run_molmo7_build():

    model_name = "allenai/Molmo-7B-D-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
    )

    tokenizer =None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def run_molmo12_build():

    model_name = "allenai/Molmo-12B-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
    )

    tokenizer =None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def molmo_inference(question, tokenizer):
    prompt = question
    return prompt


# GLM-4v: THUDM/glm-4v-9b
def run_glm4v_build():

    model_name = "THUDM/glm-4v-9b"

    llm = LLM(model=model_name,
            #   max_model_len=2048,
            #   max_num_seqs=2,
              trust_remote_code=True,
              enforce_eager=True,
              gpu_memory_utilization=0.9,
              tensor_parallel_size=2,)
    tokenizer = None
    stop_token_ids = [151329, 151336, 151338]
    return llm, tokenizer, stop_token_ids
def glm4v_inference(question, tokenizer):

    prompt = question

    return prompt


#NVLM-D-72B: nvidia/NVLM-D-72B
def run_nvlm_d_build():

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    stop_token_ids = None
    return llm, tokenizer, stop_token_ids
def nvlm_d_inference(question,tokenizer):

    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return prompt



# H2OVL-Mississippi
def run_h2ovl_build():

    model_name = "h2oai/h2ovl-mississippi-2b"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    
    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-2b
    stop_token_ids = [tokenizer.eos_token_id]
    return llm, tokenizer, stop_token_ids

def h2ovl_inference(question,tokenizer):
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt 

def run_deepseek_vl2_build():
    model_name = "deepseek-ai/deepseek-vl2-small"

    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=2,
              tensor_parallel_size=2,

            #   disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
              hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]})
    stop_token_ids = None
    tokenizer = None
    return llm, tokenizer, stop_token_ids

def deepseek_vl2_inference(question,tokenizer):
    prompt = f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"
    return prompt



model_example_map = {
    "LLaVA-1.5-7B": llava1_5_inference,
    "LLaVA-1.5-13B": llava1_5_inference,
    "LLaVA-1.6-7B": llava1_6_inference,
    "LLaVA-1.6-13B": llava1_6_inference,
    "LLaVA-1.6-34B": llava1_6_inference,
    "LLaVA-OneVision-7B": llava_onevision_inference,
    "LLaVA-OneVision-72B": llava_onevision_inference,
    "HuatuoGPT":llava_onevision_inference,
    "Fuyu-8B": fuyu_inference,
    "Phi-3-Vision": phi_inference,
    "Phi-3-5-Vision": phi_inference, 
    "PaliGemma-3B": paligemma_inference,
    "Chameleon-7B": chameleon_inference,
    "MiniCPM-V2-5": minicpmv_inference,
    "MiniCPM-V2-6": minicpmv_inference,
    "InternVL2_5-8B": internvl_inference,
    "InternVL3-8B": internvl_inference,
    "InternVL2-40B": internvl_inference,
    "InternVL-Chat-V1-5": internvl_inference,
    "BLIP-2": blip2_inference,
    "Qwen2-VL-7B": qwen2vl_inference,
    "Qwen2-VL-7B-sft": qwen2vl_inference,
    "Qwen2.5-VL-7B-sft": qwen2vl_inference,
    "MedVLM-R1": qwen2vl_inference,
    "Med-R1": qwen2vl_inference,
    "Qwen2-VL-7B-dpo": qwen2vl_inference,
    "Qwen2-VL-2B": qwen2vl_inference, 
    "Qwen2-VL-72B": qwen2vl_inference,
    "Qwen3-VL-32B": qwen2vl_inference,
    "Pixtral-12B": pixtral_hf_inference,
    "LLama 3.2-11B": mllama_inference,
    "LLama 3.2-90B": mllama_inference, 
    "Molmo-7B": molmo_inference,
    "Molmo-12B": molmo_inference, ###
    "GLM-4v": glm4v_inference,
    "NVLM-D-72B":nvlm_d_inference,
    "H2OVL-Mississippi": h2ovl_inference,
    "Deepseek-vl2-small": deepseek_vl2_inference,
    "stllava-med": llava1_5_inference,
    "Qwen2.5-VL-7B":qwen2vl_inference,
    "Lingshu-7B":qwen2vl_inference,
    "MBZUAI/MedMO-8B":qwen2vl_inference,
    "MedMO-8B":qwen2vl_inference,
    "MBZUAI/MediX-R1-8B":qwen2vl_inference,
    "MediX-R1-8B":qwen2vl_inference,
    "R1-Reward":qwen2vl_inference,
    "Qwen2.5-VL-72B":qwen2vl_inference,
    "Qwen3-VL-32B":qwen2vl_inference,
    "Qwen3-VL-8B":qwen2vl_inference,
    "LLaVA-Critic-R1-7B-Plus-Qwen":qwen2vl_inference,
}


model_llm_build = {
    "LLaVA-1.5-7B": run_llava1_5_7_build,
    "LLaVA-1.5-13B": run_llava1_5_13_build,
    "LLaVA-1.6-7B": run_llava1_6_7_build,
    "LLaVA-1.6-13B": run_llava1_6_13_build,
    "LLaVA-1.6-34B": run_llava1_6_34_build,
    "LLaVA-OneVision-7B": run_llava_onevision7_build,
    "LLaVA-OneVision-72B": run_llava_onevision72_build,
    "HuatuoGPT":run_huatuo_build,
    #"Fuyu-8B": run_fuyu_build,
    #"Phi-3-Vision": run_phi3v_build,
    "Phi-3-5-Vision":run_phi3_5v_build,
    #"PaliGemma-3B": run_paligemma_build,
    "Chameleon-7B": run_chameleon_build,
    "MiniCPM-V2-5": run_minicpmv2_5_build,
    "MiniCPM-V2-6": run_minicpmv2_6_build,
    "InternVL2_5-8B": run_internvl8b_build,
    "InternVL3-8B": run_internvl8b_build,
    "InternVL2-40B": run_internvl40b_build, 
    "InternVL-Chat-V1-5": run_internvl15_build,
    "BLIP-2": run_blip2_build,
    "Qwen2-VL-7B": run_qwen2vl7_build,
    "Qwen3-VL-32B": run_qwen3vl32_build,
    "MedVLM-R1": run_medvlmr1_build,
    "Med-R1": run_medr1_build,
    #"Qwen2-VL-2B": run_qwen2vl2_build,
    "Qwen2-VL-7B-sft": run_qwen2vl7_sft_build,
    "Qwen2.5-VL-7B-sft": run_qwen2_5vl7_sft_build,
    "Qwen2-VL-7B-dpo": run_qwen2vl7_dpo_build,
    "Qwen2-VL-72B": run_qwen2vl72_build,
    "Pixtral-12B": run_pixtral_hf_build,
    "LLama-3.2-11B": run_mllama11_build,
    #"LLama 3.2-90B": run_mllama90_build,
    "Molmo-7B": run_molmo7_build,
    #"Molmo-12B": run_molmo12_build,
    "GLM-4v": run_glm4v_build,
    #"NVLM-D-72B":run_nvlm_d_build,
    #"H2OVL-Mississippi": run_h2ovl_build,
    "Deepseek-vl2-small": run_deepseek_vl2_build,
    "stllava-med": run_stllavamed_build,
    "Qwen2.5-VL-7B":run_qwen2_5_vl7_build,
    "Lingshu-7B":run_lingshu7_build,
    "MBZUAI/MedMO-8B":run_medmo8_build,
    "MedMO-8B":run_medmo8_build,
    "MBZUAI/MediX-R1-8B":run_medix_r1_8b_build,
    "MediX-R1-8B":run_medix_r1_8b_build,
    "R1-Reward":run_r1reward_build,
    "Qwen3-VL-32B":run_qwen3_vl32_build,
    "Qwen2.5-VL-72B":run_qwen2_5_vl72_build,
    "Qwen3-VL-8B":run_qwen3_vl8_build,
    "LLaVA-Critic-R1-7B-Plus-Qwen": run_llavacritic_build,
}

def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    
        # Input image and question
    # question_file = os.path.expanduser(os.path.join(args.question_folder, f"{args.modality_type}.jsonl"))
    multi_modal_inputs=[]
    with open(question_file, 'r', encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())

            # Construct full image path
            image_path = os.path.join(entry['image_path'], entry['image_id'])

            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}. Skipping.")
                continue
            
            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            multi_modal_inputs.append({
                "image": image,
                "question": entry['question'],
                "metadata": entry
            })
        
    return multi_modal_inputs

def main(args):

    model = args.model_type   
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    
    llm, tokenizer, stop_token_ids = model_llm_build[model]()

    sampling_params = SamplingParams(temperature=args.temperature,
                                     max_tokens=4096,
                                     stop_token_ids=stop_token_ids)
    
    # modality_type = args.modality_type
    # base_image_path = os.path.expanduser(os.path.join(args.image_folder, modality_type))
    # print(base_image_path)


    ans_name = f"{args.modality_type}_{model}.jsonl"
    answers_file = os.path.expanduser(os.path.join(args.ans_folder, args.modality_type, ans_name))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # failed_name = f"{model}_failed.jsonl"
    # failed_file = os.path.expanduser(os.path.join(args.ans_path, failed_name))
    # os.makedirs(os.path.dirname(failed_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    # failed_ans_file = open(failed_file, "w")
    
    # question_file = os.path.expanduser(os.path.join(args.question_folder, f"{modality_type}.jsonl"))
    # with open(question_file, 'r', encoding="utf-8") as f:
        # lines = f.readlines()

    mm_inputs = get_multi_modal_input(args)
    # Prepare inputs for batch processing
    inputs = []
    for mm_input in mm_inputs:
        data = mm_input["image"]
        question = mm_input["question"]
        metadata = mm_input["metadata"]

        prompt = model_example_map[model](question, tokenizer)
        
        # Update sampling params with stop tokens if available
        if stop_token_ids:
            sampling_params.stop_token_ids = stop_token_ids

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": data
            },
        })


    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, (o, mm_input) in enumerate(zip(outputs, mm_inputs)):
        generated_text = o.outputs[0].text
        metadata = mm_input["metadata"]

            # print(generated_text)

        ans_file.write(json.dumps({
            "question_id": metadata.get('question_id'),
            "dataset":metadata.get('dataset'),
            "modality_type":metadata.get('modality_type'),
            "question_type": metadata.get('question_type'),
            "question": metadata.get('question'),
            "answer": generated_text,
            "image_id": metadata.get('image_id'),
            "body_part":metadata.get('body_part'),
            "answer_model": args.model_type
        }) + "\n")
        ans_file.flush()
    ans_file.close()
    # failed_ans_file.close()

def main_ori(args):

    model = args.model_type   
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    
    llm, tokenizer, stop_token_ids = model_llm_build[model]()

    sampling_params = SamplingParams(temperature=args.temperature,
                                     max_tokens=4096,
                                     stop_token_ids=stop_token_ids)
    
    modality_type = args.modality_type
    base_image_path = os.path.expanduser(os.path.join(args.image_folder, modality_type))
    # print(base_image_path)


    ans_name = f"{modality_type}_{model}.jsonl"
    answers_file = os.path.expanduser(os.path.join(args.ans_folder, modality_type, ans_name))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # failed_name = f"{model}_failed.jsonl"
    # failed_file = os.path.expanduser(os.path.join(args.ans_path, failed_name))
    # os.makedirs(os.path.dirname(failed_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    # failed_ans_file = open(failed_file, "w")
    
    question_file = os.path.expanduser(os.path.join(args.question_folder, f"{modality_type}.jsonl"))
    with open(question_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    
    for line in tqdm(lines, desc="Processing items"):
        item = json.loads(line)
        question = item['question']
        image_name = item['image_id']
        # metadata = item
        # question_id = item['question_id']
        # question_type = item['question_type']
        # data = item['dataset']
        # modality = item['modality_type']
        # body_part = item['body_part']

        # image_file = f"{base_image_path}{image_name}"
        image_file = os.path.expanduser(os.path.join(base_image_path,image_name))
        # print(image_file)
        # quit()

        image = Image.open(image_file).convert('RGB')

        prompt = model_example_map[model](question, tokenizer)

        assert args.num_prompts ==1 
        if args.num_prompts == 1:
            # Single inference
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            }
        else:
            # Batch inference
            inputs = [{
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            } for _ in range(args.num_prompts)]

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text
            # print(generated_text)

        ans_file.write(json.dumps({
            "question_id": item['question_id'],
            "dataset":item['dataset'],
            "modality_type":item['modality_type'],
            "question_type": item['question_type'],
            "question": question,
            "answer": generated_text,
            "image_id": image_name,
            "body_part":item['body_part'],
        }) + "\n")
        ans_file.flush()
    ans_file.close()
    # failed_ans_file.close()

def get_chosen_response(args, model, mm_inputs, output_path):

    # model = args.model_type   
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    
    llm, tokenizer, stop_token_ids = model_llm_build[model]()

    sampling_params = SamplingParams(temperature=args.temperature,
                                     max_tokens=4096,
                                     stop_token_ids=stop_token_ids)
    
    # modality_type = args.modality_type
    # base_image_path = os.path.expanduser(os.path.join(args.image_folder, modality_type))
    # print(base_image_path)


    # ans_name = f"{args.modality_type}_{model}.jsonl"
    # answers_file = os.path.expanduser(os.path.join(args.ans_folder, args.modality_type, ans_name))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # failed_name = f"{model}_failed.jsonl"
    # failed_file = os.path.expanduser(os.path.join(args.ans_path, failed_name))
    # os.makedirs(os.path.dirname(failed_file), exist_ok=True)

    ans_file = open(output_path, "w")
    # failed_ans_file = open(failed_file, "w")
    
    # question_file = os.path.expanduser(os.path.join(args.question_folder, f"{modality_type}.jsonl"))
    # with open(question_file, 'r', encoding="utf-8") as f:
        # lines = f.readlines()

    # mm_inputs = get_multi_modal_input(args)
    # Prepare inputs for batch processing
    inputs = []
    for mm_input in mm_inputs:
        data = mm_input["image"]
        question = mm_input["prompt"]
        metadata = mm_input["metadata"]

        prompt = model_example_map[model](question, tokenizer)
        
        # Update sampling params with stop tokens if available
        if stop_token_ids:
            sampling_params.stop_token_ids = stop_token_ids

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": data
            },
        })


    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, (o, mm_input) in enumerate(zip(outputs, mm_inputs)):
        generated_text = o.outputs[0].text
        metadata = mm_input["metadata"]

            # print(generated_text)

        ans_file.write(json.dumps({
            "question_id": metadata.get('question_id'),
            "dataset":metadata.get('dataset'),
            "modality_type":metadata.get('modality_type'),
            "question_type": metadata.get('question_type'),
            "question": metadata.get('question'),
            "chosen_answer": generated_text,
            "image_id": metadata.get('image_id'),
            "body_part":metadata.get('body_part'),
            "answer_model": args.model
        }) + "\n")
        ans_file.flush()
    ans_file.close()


