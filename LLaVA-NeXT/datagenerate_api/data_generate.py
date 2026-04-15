# import os
# import io
# import json
# import math
# import time
# import base64
# import threading
# import traceback
# from typing import List, Dict, Optional, Any
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from PIL import Image
# from openai import OpenAI
# from tqdm import tqdm


# MAX_WORKERS = 8
# API_RETRY_COUNT = 3
# API_RETRY_DELAY = 2
# MAX_PIXELS = 768 * 768

# thread_local = threading.local()
# writer_lock = threading.Lock()


# # =========================
# # OpenAI Client
# # =========================
# def get_openai_client() -> OpenAI:
#     if not hasattr(thread_local, "openai_client"):
#         thread_local.openai_client = OpenAI(
#             base_url=os.environ.get("OPENAI_BASE_URL", "http://wanqing.internal/api/gateway/v1"),
#             api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("WQ_API_KEY", "EMPTY")),
#         )
#     return thread_local.openai_client


# # =========================
# # Image Utils
# # =========================
# def resize_image_to_base64(image_path: str, max_pixels: int = MAX_PIXELS, quality: int = 85) -> str:
#     with Image.open(image_path) as img:
#         width, height = img.size
#         total_pixels = width * height

#         if total_pixels > max_pixels:
#             scale = math.sqrt(max_pixels / total_pixels)
#             new_w = max(1, int(width * scale))
#             new_h = max(1, int(height * scale))
#             img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

#         if img.mode in ("RGBA", "LA", "P"):
#             img = img.convert("RGB")

#         buffer = io.BytesIO()
#         img.save(buffer, format="JPEG", quality=quality, optimize=True)
#         buffer.seek(0)
#         return base64.b64encode(buffer.getvalue()).decode("utf-8")


# def build_multi_image_content(
#     image_paths: List[str],
#     text_prompt: str,
#     max_images: Optional[int] = None
# ) -> List[Dict[str, Any]]:
#     if max_images is not None:
#         image_paths = image_paths[:max_images]

#     content = [{"type": "text", "text": text_prompt}]
#     for img_path in image_paths:
#         img_b64 = resize_image_to_base64(img_path)
#         content.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{img_b64}"
#             }
#         })
#     return content


# # =========================
# # API Call
# # =========================
# def call_openai_api_with_retry(
#     messages: List[Dict[str, Any]],
#     model_name: str,
#     temperature: float = 0.2,
#     max_tokens: int = 2500
# ) -> Optional[str]:
#     client = get_openai_client()

#     for attempt in range(API_RETRY_COUNT):
#         try:
#             response = client.chat.completions.create(
#                 model=model_name,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#             )
#             if response and response.choices and len(response.choices) > 0:
#                 content = response.choices[0].message.content
#                 return content.strip() if content else None
#         except Exception as e:
#             print(f"[WARN] API call failed on attempt {attempt + 1}: {e}")
#             if attempt < API_RETRY_COUNT - 1:
#                 time.sleep(API_RETRY_DELAY * (attempt + 1))
#     return None


# # =========================
# # JSON Parser
# # =========================
# def safe_json_loads(text: str) -> Optional[Any]:
#     if not text:
#         return None

#     text = text.strip()

#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     # 去除 ```json ... ```
#     if text.startswith("```"):
#         text = text.strip("`")
#         text = text.replace("json\n", "", 1).strip()
#         try:
#             return json.loads(text)
#         except Exception:
#             pass

#     first_obj = text.find("{")
#     first_arr = text.find("[")

#     candidates = []
#     if first_obj != -1:
#         candidates.append(("{", first_obj))
#     if first_arr != -1:
#         candidates.append(("[", first_arr))

#     if not candidates:
#         return None

#     candidates = sorted(candidates, key=lambda x: x[1])
#     start_char, start_idx = candidates[0]
#     end_char = "}" if start_char == "{" else "]"

#     for end_idx in range(len(text) - 1, start_idx, -1):
#         if text[end_idx] == end_char:
#             substr = text[start_idx:end_idx + 1]
#             try:
#                 return json.loads(substr)
#             except Exception:
#                 continue

#     return None


# # =========================
# # Helpers
# # =========================
# def load_input_jsonl(input_path: str) -> List[Dict[str, Any]]:
#     data = []
#     with open(input_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             data.append(json.loads(line))
#     return data


# def write_jsonl(output_path: str, records: List[Dict[str, Any]]) -> None:
#     if not records:
#         return
#     with writer_lock:
#         with open(output_path, "a", encoding="utf-8") as f:
#             for record in records:
#                 f.write(json.dumps(record, ensure_ascii=False) + "\n")


# def build_abs_image_paths(image_root: str, rel_paths: List[str]) -> List[str]:
#     abs_paths = []
#     for p in rel_paths:
#         abs_p = os.path.join(image_root, p)
#         abs_paths.append(abs_p)
#     return abs_paths


# def sample_uniform_images(image_paths: List[str], max_images: int) -> List[str]:
#     if len(image_paths) <= max_images:
#         return image_paths
#     if max_images <= 1:
#         return [image_paths[0]]

#     idxs = [round(i * (len(image_paths) - 1) / (max_images - 1)) for i in range(max_images)]
#     idxs = sorted(set(idxs))
#     return [image_paths[i] for i in idxs]


# def build_reference_qa(conversation: List[Dict[str, str]], max_turns: int = 8) -> str:
#     lines = []
#     cnt = 0
#     for item in conversation:
#         role = item.get("role", "")
#         content = item.get("content", "").strip()
#         if not content:
#             continue
#         lines.append(f"{role}: {content}")
#         cnt += 1
#         if cnt >= max_turns:
#             break
#     return "\n".join(lines)


# # =========================
# # Prompt Builders
# # =========================
# def build_stage1_analysis_prompt(sample_id: str, source: str, conversation: List[Dict[str, str]], image_rel_paths: List[str]) -> str:
#     ref_qa = build_reference_qa(conversation, max_turns=8)
#     image_index = "\n".join([f"image{i+1}: {p}" for i, p in enumerate(image_rel_paths)])

#     return f"""
# You are analyzing a multi-image set for dataset construction.

# Dataset source: {source}
# Sample ID: {sample_id}

# Image index mapping:
# {image_index}

# Optional reference QA from the original dataset (for context only; do not blindly trust it over the images):
# {ref_qa}

# Given the images, do NOT answer any downstream question yet.
# Instead, analyze the image set in a structured way.

# Please output valid JSON with the following fields:
# {{
#   "per_image_summary": {{
#     "image1": "...",
#     "image2": "..."
#   }},
#   "shared_elements": ["...", "..."],
#   "cross_image_differences": ["...", "..."],
#   "possible_global_inferences": [
#     {{
#       "inference": "...",
#       "confidence": "high/medium/low"
#     }}
#   ],
#   "possible_evidence_paths": [
#     {{
#       "inference": "...",
#       "path_a": {{
#         "images": ["image1", "image3"],
#         "cues": ["...", "..."]
#       }},
#       "path_b": {{
#         "images": ["image2", "image4"],
#         "cues": ["...", "..."]
#       }},
#       "independent_support": "yes/no/uncertain"
#     }}
#   ],
#   "multi_image_requirement": {{
#     "is_multi_image_necessary": true,
#     "reason": "..."
#   }}
# }}

# Requirements:
# - Stay factual and grounded in the images.
# - Avoid hallucinations.
# - If uncertain, explicitly say uncertain.
# - Prefer conclusions that are visible or strongly supported.
# - Return JSON only.
# """.strip()


# def build_stage2_question_generation_prompt(
#     sample_id: str,
#     source: str,
#     conversation: List[Dict[str, str]],
#     analysis: Dict[str, Any]
# ) -> str:
#     ref_qa = build_reference_qa(conversation, max_turns=8)

#     return f"""
# You are constructing multi-image questions for preference learning research.

# Sample ID: {sample_id}
# Dataset source: {source}

# Original conversation (reference only):
# {ref_qa}

# Based on the structured image-set analysis below, generate candidate questions that satisfy all of the following:

# 1. The question should genuinely require multiple images.
# 2. The answer should be relatively determinate or well-supported.
# 3. The supporting evidence path should not be unique: different subsets of images and/or different visual cues may justify the same answer.
# 4. The question should allow multiple equally valid grounded responses that differ in explanation style, evidence choice, or level of detail.
# 5. Avoid overly subjective, purely aesthetic, or trivial questions.

# Generate 8 candidate questions total, with the following distribution:
# - 3 aggregation/inference questions
# - 2 comparison or grouping questions
# - 2 temporal/change questions
# - 1 supportability/evidence question

# For each question, provide:
# - question_id
# - question_text
# - question_type
# - why_multi_image
# - expected_answer_form
# - why_tie_likely
# - confidence_that_question_fits

# Return valid JSON only as a list.

# Structured image-set analysis:
# {json.dumps(analysis, ensure_ascii=False, indent=2)}
# """.strip()


# def build_stage3_question_filter_prompt(
#     sample_id: str,
#     source: str,
#     question_candidates: List[Dict[str, Any]]
# ) -> str:
#     return f"""
# You are selecting high-quality multi-image questions for a tie-aware preference dataset.

# Sample ID: {sample_id}
# Dataset source: {source}

# Below are candidate questions for one image set. Evaluate each question using the following criteria:

# A. Multi-image necessity:
# Does the question truly require multiple images?

# B. Answer determinacy:
# Is there a relatively well-supported answer, rather than a purely subjective one?

# C. Non-unique support:
# Can the same answer plausibly be justified by different image subsets or different visual cues?

# D. Tie potential:
# Would this question naturally allow multiple equally valid grounded responses that are hard to strictly rank?

# E. Dataset usefulness:
# Would this question be useful for studying non-strict preferences in multi-image MLLMs?

# For each question:
# - score each criterion from 1 to 5
# - provide a brief rationale
# - decide keep or discard

# Then select the best 1 or 2 questions and rewrite them into clearer final versions.

# Return JSON only in the format:
# {{
#   "evaluations": [...],
#   "selected_questions": [
#     {{
#       "question_id": "...",
#       "final_question_text": "...",
#       "question_type": "...",
#       "selection_reason": "..."
#     }}
#   ]
# }}

# Candidate questions:
# {json.dumps(question_candidates, ensure_ascii=False, indent=2)}
# """.strip()


# def build_stage4_response_generation_prompt(
#     sample_id: str,
#     source: str,
#     selected_question: Dict[str, Any],
#     analysis: Dict[str, Any]
# ) -> str:
#     return f"""
# You are generating candidate responses to a multi-image question.

# Sample ID: {sample_id}
# Dataset source: {source}

# Selected question:
# {json.dumps(selected_question, ensure_ascii=False, indent=2)}

# Structured image analysis:
# {json.dumps(analysis, ensure_ascii=False, indent=2)}

# Given the images and the selected question, produce 8 candidate answers that are all grounded in the image set.

# The responses should differ along these dimensions:
# 1. evidence choice: cite different valid visual cues and/or different image subsets when possible
# 2. response style: concise vs detailed
# 3. organization: conclusion-first vs evidence-first
# 4. level of explanation: minimal sufficient explanation vs fuller explanation

# Constraints:
# - 6 responses should be high-quality grounded responses.
# - 2 responses should be borderline weaker but still plausible (e.g. less complete, less grounded, slightly vague), but not clearly wrong.
# - All responses must remain faithful to the images.
# - Do not intentionally make any response false.
# - If the image evidence is insufficient for some style variation, keep the response conservative.
# - Avoid superficial paraphrases; make the differences meaningful.

# Return valid JSON only as a list, where each item has:
# {{
#   "response_id": "r1",
#   "answer_text": "...",
#   "claimed_conclusion": "...",
#   "evidence_used": {{
#     "images": ["image1", "image3"],
#     "cues": ["...", "..."]
#   }},
#   "style_tags": ["concise", "conclusion_first"],
#   "confidence": "high/medium/low",
#   "quality_band": "high/borderline"
# }}
# """.strip()


# # =========================
# # Generation Functions
# # =========================
# def run_multimodal_json_stage(
#     image_paths: List[str],
#     prompt: str,
#     model_name: str,
#     max_images: int = 8,
#     temperature: float = 0.2,
#     max_tokens: int = 2500
# ) -> Optional[Any]:
#     content = build_multi_image_content(image_paths, prompt, max_images=max_images)
#     messages = [
#         {
#             "role": "system",
#             "content": "You are an AI assistant for structured multi-image dataset construction. Return valid JSON only."
#         },
#         {
#             "role": "user",
#             "content": content
#         }
#     ]
#     response = call_openai_api_with_retry(
#         messages=messages,
#         model_name=model_name,
#         temperature=temperature,
#         max_tokens=max_tokens
#     )
#     return safe_json_loads(response)


# def generate_stage1_analysis(
#     image_paths: List[str],
#     sample_id: str,
#     source: str,
#     conversation: List[Dict[str, str]],
#     image_rel_paths: List[str],
#     model_name: str,
#     max_images: int
# ) -> Dict[str, Any]:
#     prompt = build_stage1_analysis_prompt(sample_id, source, conversation, image_rel_paths)
#     result = run_multimodal_json_stage(
#         image_paths=image_paths,
#         prompt=prompt,
#         model_name=model_name,
#         max_images=max_images,
#         temperature=0.1,
#         max_tokens=2200
#     )
#     if isinstance(result, dict):
#         return result
#     return {
#         "per_image_summary": {},
#         "shared_elements": [],
#         "cross_image_differences": [],
#         "possible_global_inferences": [],
#         "possible_evidence_paths": [],
#         "multi_image_requirement": {
#             "is_multi_image_necessary": None,
#             "reason": ""
#         }
#     }


# def generate_stage2_questions(
#     image_paths: List[str],
#     sample_id: str,
#     source: str,
#     conversation: List[Dict[str, str]],
#     analysis: Dict[str, Any],
#     model_name: str,
#     max_images: int
# ) -> List[Dict[str, Any]]:
#     prompt = build_stage2_question_generation_prompt(sample_id, source, conversation, analysis)
#     result = run_multimodal_json_stage(
#         image_paths=image_paths,
#         prompt=prompt,
#         model_name=model_name,
#         max_images=max_images,
#         temperature=0.4,
#         max_tokens=2200
#     )
#     if isinstance(result, list):
#         return [x for x in result if isinstance(x, dict)]
#     return []


# def generate_stage3_selection(
#     image_paths: List[str],
#     sample_id: str,
#     source: str,
#     question_candidates: List[Dict[str, Any]],
#     model_name: str,
#     max_images: int
# ) -> Dict[str, Any]:
#     prompt = build_stage3_question_filter_prompt(sample_id, source, question_candidates)
#     result = run_multimodal_json_stage(
#         image_paths=image_paths,
#         prompt=prompt,
#         model_name=model_name,
#         max_images=max_images,
#         temperature=0.2,
#         max_tokens=2200
#     )
#     if isinstance(result, dict):
#         return result
#     return {
#         "evaluations": [],
#         "selected_questions": []
#     }


# def generate_stage4_responses(
#     image_paths: List[str],
#     sample_id: str,
#     source: str,
#     selected_question: Dict[str, Any],
#     analysis: Dict[str, Any],
#     model_name: str,
#     max_images: int
# ) -> List[Dict[str, Any]]:
#     prompt = build_stage4_response_generation_prompt(sample_id, source, selected_question, analysis)
#     result = run_multimodal_json_stage(
#         image_paths=image_paths,
#         prompt=prompt,
#         model_name=model_name,
#         max_images=max_images,
#         temperature=0.7,
#         max_tokens=3000
#     )
#     if isinstance(result, list):
#         return [x for x in result if isinstance(x, dict)]
#     return []


# # =========================
# # Core Processing
# # =========================
# def process_sample(
#     sample: Dict[str, Any],
#     image_root: str,
#     model_name: str,
#     max_images: int = 8
# ) -> List[Dict[str, Any]]:
#     sample_id = sample["id"]
#     image_rel_paths = sample["images"]
#     conversation = sample.get("conversation", [])
#     source = sample.get("source", "unknown")

#     if not image_rel_paths:
#         return []

#     # 拼接绝对路径
#     image_abs_paths = build_abs_image_paths(image_root, image_rel_paths)

#     # 检查图片存在
#     valid_rel_paths = []
#     valid_abs_paths = []
#     for rel_p, abs_p in zip(image_rel_paths, image_abs_paths):
#         if os.path.exists(abs_p):
#             valid_rel_paths.append(rel_p)
#             valid_abs_paths.append(abs_p)

#     if not valid_abs_paths:
#         return []

#     # 均匀采样图片，避免故事序列只截前几张
#     paired = list(zip(valid_rel_paths, valid_abs_paths))
#     if len(paired) > max_images:
#         idxs = [round(i * (len(paired) - 1) / (max_images - 1)) for i in range(max_images)]
#         idxs = sorted(set(idxs))
#         paired = [paired[i] for i in idxs]

#     used_rel_paths = [x[0] for x in paired]
#     used_abs_paths = [x[1] for x in paired]

#     # Stage 1
#     analysis = generate_stage1_analysis(
#         image_paths=used_abs_paths,
#         sample_id=sample_id,
#         source=source,
#         conversation=conversation,
#         image_rel_paths=used_rel_paths,
#         model_name=model_name,
#         max_images=max_images
#     )

#     # Stage 2
#     question_candidates = generate_stage2_questions(
#         image_paths=used_abs_paths,
#         sample_id=sample_id,
#         source=source,
#         conversation=conversation,
#         analysis=analysis,
#         model_name=model_name,
#         max_images=max_images
#     )

#     # Stage 3
#     question_selection = generate_stage3_selection(
#         image_paths=used_abs_paths,
#         sample_id=sample_id,
#         source=source,
#         question_candidates=question_candidates,
#         model_name=model_name,
#         max_images=max_images
#     )

#     selected_questions = question_selection.get("selected_questions", [])
#     if not selected_questions:
#         return [{
#             "sample_id": sample_id,
#             "source": source,
#             "images": used_rel_paths,
#             "original_conversation": conversation,
#             "stage1_analysis": analysis,
#             "stage2_question_candidates": question_candidates,
#             "stage3_question_selection": question_selection,
#             "stage4_responses": [],
#             "status": "no_selected_question"
#         }]

#     outputs = []
#     for s_idx, selected_question in enumerate(selected_questions[:2]):
#         responses = generate_stage4_responses(
#             image_paths=used_abs_paths,
#             sample_id=sample_id,
#             source=source,
#             selected_question=selected_question,
#             analysis=analysis,
#             model_name=model_name,
#             max_images=max_images
#         )

#         outputs.append({
#             "sample_id": sample_id,
#             "record_id": f"{sample_id}_sel{s_idx}",
#             "source": source,
#             "images": used_rel_paths,
#             "original_conversation": conversation,
#             "stage1_analysis": analysis,
#             "stage2_question_candidates": question_candidates,
#             "stage3_question_selection": question_selection,
#             "selected_question": selected_question,
#             "stage4_responses": responses,
#             "status": "ok"
#         })

#     return outputs


# def process_and_write_sample(
#     sample: Dict[str, Any],
#     output_path: str,
#     image_root: str,
#     model_name: str,
#     max_images: int
# ) -> int:
#     try:
#         records = process_sample(
#             sample=sample,
#             image_root=image_root,
#             model_name=model_name,
#             max_images=max_images
#         )
#         write_jsonl(output_path, records)
#         return len(records)
#     except Exception as e:
#         err_record = {
#             "sample_id": sample.get("id", "unknown"),
#             "source": sample.get("source", "unknown"),
#             "images": sample.get("images", []),
#             "status": "error",
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }
#         write_jsonl(output_path, [err_record])
#         print(f"[ERROR] Failed sample {sample.get('id', 'unknown')}: {e}")
#         return 0


# # =========================
# # Main
# # =========================
# def main():
#     import argparse

#     parser = argparse.ArgumentParser(description="Multi-image Stage1-4 pipeline")

#     parser.add_argument(
#         "--input",
#         type=str,
#         default="/home/kuai-blm/zhangkeyao/mantis/mantis_export/visual_story_telling/train/annotations.jsonl",
#         help="Input JSONL path"
#     )
#     parser.add_argument(
#         "--image-root",
#         type=str,
#         default="/home/kuai-blm/zhangkeyao/mantis/mantis_export/visual_story_telling/train",
#         help="Image root directory"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="./stage1_4_test_output",
#         help="Directory to save output"
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="ep-ncdc7s-1775636894248361307",
#         help="Model name for API"
#     )
#     parser.add_argument(
#         "--max-workers",
#         type=int,
#         default=4,
#         help="Number of workers"
#     )
#     parser.add_argument(
#         "--max-images",
#         type=int,
#         default=6,
#         help="Maximum number of images per sample"
#     )
#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=10,
#         help="Maximum number of samples to process"
#     )

#     args = parser.parse_args()

#     print("========== Running Config ==========")
#     print(f"input       : {args.input}")
#     print(f"image_root  : {args.image_root}")
#     print(f"output_dir  : {args.output_dir}")
#     print(f"model       : {args.model}")
#     print(f"max_workers : {args.max_workers}")
#     print(f"max_images  : {args.max_images}")
#     print(f"limit       : {args.limit}")
#     print("====================================")

#     os.makedirs(args.output_dir, exist_ok=True)
#     output_path = os.path.join(args.output_dir, "stage1_4_results.jsonl")

#     samples = load_input_jsonl(args.input)
#     if args.limit > 0:
#         samples = samples[:args.limit]

#     print(f"Loaded {len(samples)} samples")
#     print(f"Saving to: {output_path}")
#     print(f"Using model: {args.model}")
#     print(f"Image root: {args.image_root}")

#     total_records = 0
#     with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
#         futures = [
#             executor.submit(
#                 process_and_write_sample,
#                 sample,
#                 output_path,
#                 args.image_root,
#                 args.model,
#                 args.max_images
#             )
#             for sample in samples
#         ]

#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
#             total_records += future.result()

#     print(f"Done. Generated {total_records} records.")


# if __name__ == "__main__":
#     main()


import os
import io
import json
import math
import time
import base64
import threading
import traceback
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from openai import OpenAI
from tqdm import tqdm


MAX_WORKERS = 4
API_RETRY_COUNT = 3
API_RETRY_DELAY = 2
MAX_PIXELS = 768 * 768

thread_local = threading.local()
writer_lock = threading.Lock()


# =========================
# OpenAI Client
# =========================
def get_openai_client() -> OpenAI:
    if not hasattr(thread_local, "openai_client"):
        thread_local.openai_client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://wanqing.internal/api/gateway/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("WQ_API_KEY", "EMPTY")),
        )
    return thread_local.openai_client


# =========================
# Image Utils
# =========================
def resize_image_to_base64(image_path: str, max_pixels: int = MAX_PIXELS, quality: int = 85) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
        total_pixels = width * height

        if total_pixels > max_pixels:
            scale = math.sqrt(max_pixels / total_pixels)
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_multi_image_content(
    image_paths: List[str],
    text_prompt: str,
    max_images: Optional[int] = None
) -> List[Dict[str, Any]]:
    if max_images is not None:
        image_paths = image_paths[:max_images]

    content = [{"type": "text", "text": text_prompt}]
    for img_path in image_paths:
        img_b64 = resize_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })
    return content


# =========================
# API Call
# =========================
def call_openai_api_with_retry(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 1800
) -> Optional[str]:
    client = get_openai_client()

    for attempt in range(API_RETRY_COUNT):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content.strip() if content else None
        except Exception as e:
            print(f"[WARN] API call failed on attempt {attempt + 1}: {e}")
            if attempt < API_RETRY_COUNT - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))
    return None


# =========================
# JSON Parser
# =========================
def safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
        try:
            return json.loads(text)
        except Exception:
            pass

    first_obj = text.find("{")
    first_arr = text.find("[")

    candidates = []
    if first_obj != -1:
        candidates.append(("{", first_obj))
    if first_arr != -1:
        candidates.append(("[", first_arr))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x[1])
    start_char, start_idx = candidates[0]
    end_char = "}" if start_char == "{" else "]"

    for end_idx in range(len(text) - 1, start_idx, -1):
        if text[end_idx] == end_char:
            substr = text[start_idx:end_idx + 1]
            try:
                return json.loads(substr)
            except Exception:
                continue

    return None


# =========================
# IO / Helpers
# =========================
def load_input_jsonl(input_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(output_path: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with writer_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_abs_image_paths(image_root: str, rel_paths: List[str]) -> List[str]:
    return [os.path.join(image_root, p) for p in rel_paths]


def build_reference_qa(conversation: List[Dict[str, str]], max_turns: int = 6) -> str:
    lines = []
    cnt = 0
    for item in conversation:
        role = item.get("role", "")
        content = item.get("content", "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
        cnt += 1
        if cnt >= max_turns:
            break
    return "\n".join(lines)


def sample_uniform_pairs(rel_paths: List[str], abs_paths: List[str], max_images: int) -> Tuple[List[str], List[str]]:
    paired = list(zip(rel_paths, abs_paths))
    if len(paired) <= max_images:
        return rel_paths, abs_paths
    if max_images <= 1:
        return [rel_paths[0]], [abs_paths[0]]

    idxs = [round(i * (len(paired) - 1) / (max_images - 1)) for i in range(max_images)]
    idxs = sorted(set(idxs))
    paired = [paired[i] for i in idxs]
    return [x[0] for x in paired], [x[1] for x in paired]


# =========================
# Prompt Builders
# =========================
def build_best_question_prompt(
    sample_id: str,
    source: str,
    conversation: List[Dict[str, str]],
    image_rel_paths: List[str]
) -> str:
    ref_qa = build_reference_qa(conversation, max_turns=6)
    image_index = "\n".join([f"image{i+1}: {p}" for i, p in enumerate(image_rel_paths)])

    return f"""
You are constructing low-cost but high-quality multi-image preference-learning data.

Dataset source: {source}
Sample ID: {sample_id}

Image index mapping:
{image_index}

Original conversation from dataset (reference only, do not blindly trust it over the images):
{ref_qa}

Your task:
Look at the image set and produce ONE best question for multi-image preference learning.

The question must satisfy:
1. It genuinely requires multiple images.
2. It should have a relatively supported answer, not a purely subjective one.
3. The same answer could be supported by different visual cues or different subsets of images.
4. It should naturally allow multiple equally valid grounded responses that differ in style, evidence choice, or detail level.
5. Keep the question natural and concise.

Return valid JSON only in this format:
{{
  "best_question": "...",
  "question_type": "aggregation/comparison/temporal/supportability/summary",
  "why_multi_image": "...",
  "why_tie_likely": "...",
  "concise_scene_summary": "1-2 sentence factual summary of the whole image set"
}}
""".strip()


def build_response_generation_prompt(
    sample_id: str,
    source: str,
    question_info: Dict[str, Any]
) -> str:
    return f"""
You are generating candidate responses to a multi-image question.

Dataset source: {source}
Sample ID: {sample_id}

Question info:
{json.dumps(question_info, ensure_ascii=False, indent=2)}

Given the images and the question, generate 6 candidate responses.

Requirements:
- 4 responses should be high-quality, grounded, and mutually diverse.
- 2 responses should be slightly weaker or more borderline, but still plausible and not clearly wrong.
- Responses should differ meaningfully in:
  1. evidence choice
  2. response style
  3. detail level
  4. organization
- Avoid trivial paraphrases.
- Do not intentionally introduce hallucinations.

Return valid JSON only as a list. Each item must contain:
{{
  "response_id": "r1",
  "answer_text": "...",
  "claimed_conclusion": "...",
  "evidence_used": {{
    "images": ["image1", "image3"],
    "cues": ["...", "..."]
  }},
  "style_tags": ["concise", "conclusion_first"],
  "quality_band": "high/borderline"
}}
""".strip()


def build_pair_judge_prompt(
    question_info: Dict[str, Any],
    response_a: Dict[str, Any],
    response_b: Dict[str, Any]
) -> str:
    return f"""
You are judging two candidate responses to the same multi-image question.

Question info:
{json.dumps(question_info, ensure_ascii=False, indent=2)}

Response A:
{json.dumps(response_a, ensure_ascii=False, indent=2)}

Response B:
{json.dumps(response_b, ensure_ascii=False, indent=2)}

Assign exactly one label:
- better_a
- better_b
- tie

Definition of tie:
Choose tie when both responses are similarly good in correctness, grounding, relevance, and adequacy, and any difference is mainly due to style, detail level, organization, or alternative but equally valid evidence choice.

Priority of evaluation:
1. correctness of conclusion
2. faithfulness / grounding
3. relevance to question
4. adequacy / completeness
5. clarity
6. conciseness only as a minor factor

Important:
- Do not force a strict winner if the two are effectively equivalent.
- If both responses support the same conclusion using different valid cues, tie is often appropriate.
- If one is only slightly more detailed but the other is already sufficient, this may still be tie.
- If one is vaguer, less grounded, or less adequate, then prefer the stronger one.

Return valid JSON only in this format:
{{
  "label": "better_a/better_b/tie",
  "confidence": "high/medium/low",
  "rationale": "...",
  "tie_type": "evidence_path/style_variation/detail_level/other/none",
  "is_strict_preference_reliable": "yes/no"
}}
""".strip()


# =========================
# Model Runners
# =========================
def run_multimodal_json(
    image_paths: List[str],
    prompt: str,
    model_name: str,
    max_images: int,
    temperature: float,
    max_tokens: int
) -> Optional[Any]:
    content = build_multi_image_content(image_paths, prompt, max_images=max_images)
    messages = [
        {"role": "system", "content": "You are an AI assistant for structured multi-image dataset construction. Return valid JSON only."},
        {"role": "user", "content": content}
    ]
    raw = call_openai_api_with_retry(
        messages=messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return safe_json_loads(raw)


def run_text_json(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int
) -> Optional[Any]:
    messages = [
        {"role": "system", "content": "You are an AI assistant for structured preference annotation. Return valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    raw = call_openai_api_with_retry(
        messages=messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return safe_json_loads(raw)


# =========================
# Generation
# =========================
def generate_best_question(
    image_paths: List[str],
    sample_id: str,
    source: str,
    conversation: List[Dict[str, str]],
    image_rel_paths: List[str],
    model_name: str,
    max_images: int
) -> Dict[str, Any]:
    prompt = build_best_question_prompt(sample_id, source, conversation, image_rel_paths)
    result = run_multimodal_json(
        image_paths=image_paths,
        prompt=prompt,
        model_name=model_name,
        max_images=max_images,
        temperature=0.2,
        max_tokens=1200
    )
    if isinstance(result, dict) and result.get("best_question"):
        return result

    return {
        "best_question": "",
        "question_type": "unknown",
        "why_multi_image": "",
        "why_tie_likely": "",
        "concise_scene_summary": ""
    }


def generate_candidate_responses(
    image_paths: List[str],
    sample_id: str,
    source: str,
    question_info: Dict[str, Any],
    model_name: str,
    max_images: int
) -> List[Dict[str, Any]]:
    prompt = build_response_generation_prompt(sample_id, source, question_info)
    result = run_multimodal_json(
        image_paths=image_paths,
        prompt=prompt,
        model_name=model_name,
        max_images=max_images,
        temperature=0.7,
        max_tokens=1800
    )
    if isinstance(result, list):
        return [x for x in result if isinstance(x, dict) and x.get("answer_text")]
    return []


# =========================
# Pair Construction
# =========================
def build_informative_pairs(responses: List[Dict[str, Any]], max_pairs: int = 5) -> List[Dict[str, Any]]:
    """
    从 6 个回答中构造最有信息量的 pair，尽量避免全组合。
    策略：
    - high vs high: 2 对
    - high vs borderline: 2 对
    - 不同 style / evidence: 1 对
    """
    highs = [r for r in responses if r.get("quality_band") == "high"]
    borders = [r for r in responses if r.get("quality_band") == "borderline"]

    pairs = []
    used = set()

    def add_pair(a: Dict[str, Any], b: Dict[str, Any], pair_type: str):
        if not a or not b:
            return
        ra = a.get("response_id")
        rb = b.get("response_id")
        if not ra or not rb or ra == rb:
            return
        key = tuple(sorted([ra, rb]))
        if key in used:
            return
        used.add(key)
        pairs.append({
            "pair_id": f"{ra}__{rb}",
            "response_a_id": ra,
            "response_b_id": rb,
            "pair_type": pair_type
        })

    # 1) high vs high
    for i in range(min(len(highs), 3)):
        for j in range(i + 1, min(len(highs), 4)):
            if len([p for p in pairs if p["pair_type"] == "high_high"]) < 2:
                add_pair(highs[i], highs[j], "high_high")

    # 2) high vs borderline
    for h in highs[:2]:
        for b in borders[:2]:
            if len([p for p in pairs if p["pair_type"] == "high_borderline"]) < 2:
                add_pair(h, b, "high_borderline")

    # 3) extra diverse pair
    if len(highs) >= 2:
        add_pair(highs[-1], highs[0], "diverse_high")

    return pairs[:max_pairs]


def index_responses_by_id(responses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["response_id"]: r for r in responses if r.get("response_id")}


def judge_pair(
    question_info: Dict[str, Any],
    response_a: Dict[str, Any],
    response_b: Dict[str, Any],
    model_name: str
) -> Dict[str, Any]:
    prompt = build_pair_judge_prompt(question_info, response_a, response_b)
    result = run_text_json(
        prompt=prompt,
        model_name=model_name,
        temperature=0.1,
        max_tokens=600
    )
    if isinstance(result, dict) and result.get("label"):
        return result

    return {
        "label": "tie",
        "confidence": "low",
        "rationale": "fallback due to parse failure",
        "tie_type": "other",
        "is_strict_preference_reliable": "no"
    }


# =========================
# Core Processing
# =========================
def process_sample(
    sample: Dict[str, Any],
    image_root: str,
    model_name: str,
    max_images: int = 4,
    max_pairs: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sample_id = sample.get("id", "")
    image_rel_paths = sample.get("images", [])
    conversation = sample.get("conversation", [])
    source = sample.get("source", "unknown")

    if not image_rel_paths:
        return [], []

    image_abs_paths = build_abs_image_paths(image_root, image_rel_paths)

    valid_rel_paths = []
    valid_abs_paths = []
    for rel_p, abs_p in zip(image_rel_paths, image_abs_paths):
        if os.path.exists(abs_p):
            valid_rel_paths.append(rel_p)
            valid_abs_paths.append(abs_p)

    if not valid_abs_paths:
        return [], []

    used_rel_paths, used_abs_paths = sample_uniform_pairs(valid_rel_paths, valid_abs_paths, max_images=max_images)

    # Step A: best question
    question_info = generate_best_question(
        image_paths=used_abs_paths,
        sample_id=sample_id,
        source=source,
        conversation=conversation,
        image_rel_paths=used_rel_paths,
        model_name=model_name,
        max_images=max_images
    )

    if not question_info.get("best_question"):
        question_record = {
            "sample_id": sample_id,
            "source": source,
            "images": used_rel_paths,
            "original_conversation": conversation,
            "question_info": question_info,
            "responses": [],
            "status": "no_question"
        }
        return [question_record], []

    # Step B: responses
    responses = generate_candidate_responses(
        image_paths=used_abs_paths,
        sample_id=sample_id,
        source=source,
        question_info=question_info,
        model_name=model_name,
        max_images=max_images
    )

    question_record = {
        "sample_id": sample_id,
        "source": source,
        "images": used_rel_paths,
        "original_conversation": conversation,
        "question_info": question_info,
        "responses": responses,
        "status": "ok" if responses else "no_responses"
    }

    if not responses:
        return [question_record], []

    # Step C: build pairs
    pair_specs = build_informative_pairs(responses, max_pairs=max_pairs)
    resp_map = index_responses_by_id(responses)

    # Step D: judge pairs (text only)
    pair_records = []
    for spec in pair_specs:
        ra = resp_map.get(spec["response_a_id"])
        rb = resp_map.get(spec["response_b_id"])
        if not ra or not rb:
            continue

        judge_result = judge_pair(question_info, ra, rb, model_name=model_name)

        pair_records.append({
            "sample_id": sample_id,
            "source": source,
            "images": used_rel_paths,
            "question": question_info.get("best_question", ""),
            "question_type": question_info.get("question_type", ""),
            "concise_scene_summary": question_info.get("concise_scene_summary", ""),
            "pair_id": spec["pair_id"],
            "pair_type": spec["pair_type"],
            "response_a": ra,
            "response_b": rb,
            "label": judge_result.get("label", "tie"),
            "confidence": judge_result.get("confidence", "low"),
            "rationale": judge_result.get("rationale", ""),
            "tie_type": judge_result.get("tie_type", "none"),
            "is_strict_preference_reliable": judge_result.get("is_strict_preference_reliable", "no")
        })

    return [question_record], pair_records


def process_and_write_sample(
    sample: Dict[str, Any],
    qr_output_path: str,
    pair_output_path: str,
    image_root: str,
    model_name: str,
    max_images: int,
    max_pairs: int
) -> Tuple[int, int]:
    try:
        question_records, pair_records = process_sample(
            sample=sample,
            image_root=image_root,
            model_name=model_name,
            max_images=max_images,
            max_pairs=max_pairs
        )
        write_jsonl(qr_output_path, question_records)
        write_jsonl(pair_output_path, pair_records)
        return len(question_records), len(pair_records)
    except Exception as e:
        err_record = {
            "sample_id": sample.get("id", "unknown"),
            "source": sample.get("source", "unknown"),
            "images": sample.get("images", []),
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        write_jsonl(qr_output_path, [err_record])
        print(f"[ERROR] Failed sample {sample.get('id', 'unknown')}: {e}")
        return 1, 0


# =========================
# Main
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Low-token multi-image tie data pipeline")

    parser.add_argument(
        "--input",
        type=str,
        default="/home/kuai-blm/zhangkeyao/mantis/mantis_export/visual_story_telling/train/annotations.jsonl",
        help="Input JSONL path"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/home/kuai-blm/zhangkeyao/mantis/mantis_export/visual_story_telling/train",
        help="Image root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./low_token_tie_output",
        help="Directory to save output"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ep-ncdc7s-1775636894248361307",
        help="Model name for API"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=6,
        help="Maximum number of images per sample"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=5,
        help="Maximum number of judged pairs per sample"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of samples to process"
    )

    args = parser.parse_args()

    print("========== Running Config ==========")
    print(f"input       : {args.input}")
    print(f"image_root  : {args.image_root}")
    print(f"output_dir  : {args.output_dir}")
    print(f"model       : {args.model}")
    print(f"max_workers : {args.max_workers}")
    print(f"max_images  : {args.max_images}")
    print(f"max_pairs   : {args.max_pairs}")
    print(f"limit       : {args.limit}")
    print("====================================")

    os.makedirs(args.output_dir, exist_ok=True)

    qr_output_path = os.path.join(
        args.output_dir,
        f"question_response_records_limit{args.limit}_img{args.max_images}.jsonl"
    )
    pair_output_path = os.path.join(
        args.output_dir,
        f"pair_pref_records_limit{args.limit}_img{args.max_images}.jsonl"
    )

    samples = load_input_jsonl(args.input)
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Loaded {len(samples)} samples")
    print(f"Question/response output: {qr_output_path}")
    print(f"Pair preference output  : {pair_output_path}")

    total_qr = 0
    total_pairs = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_and_write_sample,
                sample,
                qr_output_path,
                pair_output_path,
                args.image_root,
                args.model,
                args.max_images,
                args.max_pairs
            )
            for sample in samples
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            qr_n, pair_n = future.result()
            total_qr += qr_n
            total_pairs += pair_n

    print(f"Done. question_records={total_qr}, pair_records={total_pairs}")


if __name__ == "__main__":
    main()






