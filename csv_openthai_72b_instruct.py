import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder
import pytrec_eval

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

model_name = "openthaigpt/openthaigpt-1.6-72b-instruct"
token = HfFolder.get_token()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=token
)

# ฟังก์ชันถามคำถามกฎหมาย
def ask_law_question(question, context):
    messages = [
        {"role": "system", "content": """ 
        คุณคือนักกฎหมายผู้เชี่ยวชาญกฎหมายแพ่งและพาณิชย์
        จงอ่านข้อความคดีต่อไปนี้อย่างละเอียด แล้วตอบเฉพาะ "เลขมาตรากฎหมายแพ่งและพาณิชย์" ที่เกี่ยวข้องมากที่สุด 5 มาตราแรก ที่งเกี่ยวข้องโดยตรงเเละโดยอ้อม โดยเรียงตามลำดับความเกี่ยวข้องมากไปน้อย ห้ามเกิน 5 มาตรา และไม่ต้องมีคำว่า "มาตรา"
        ให้ตอบเป็นเลขคั่นด้วยเครื่องหมายจุลภาค เช่น: 1336, 1299, 1520, 1500, 1337
        """},
        {"role": "user", "content": f"""
        ข้อความ:\n{context}\n\nคำถาม:\n{question}
        """},
    ]


    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# ฟังก์ชันแยกเฉพาะเลขมาตรา เช่น 1336
def extract_law_numbers(text):
    return re.findall(r'\d+', text)

def evaluate_with_pytrec(df, k=5):
    qrel = {}
    run = {}

    for idx, row in df.iterrows():
        qid = f"q{idx}"
        true_ids = extract_law_numbers(str(row['answers']))
        pred_ids = extract_law_numbers(str(row['predicted_law']))[:k]

        qrel[qid] = {law_id: 1 for law_id in true_ids}
        run[qid] = {law_id: 1.0 / (i + 1) for i, law_id in enumerate(pred_ids)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"recip_rank", f"recall_{k}"})
    results = evaluator.evaluate(run)

    df["recall_at_k"] = 0.0
    df["mrr_at_k"] = 0.0
    recall_scores = []
    mrr_scores = []

    for qid, metrics in results.items():
        idx = int(qid[1:])
        recall = metrics.get(f"recall_{k}", 0.0)
        mrr = metrics.get("recip_rank", 0.0)

        df.at[idx, "recall_at_k"] = recall
        df.at[idx, "mrr_at_k"] = mrr

        recall_scores.append(recall)
        mrr_scores.append(mrr)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    return avg_recall, avg_mrr

# Main process
if __name__ == "__main__":
    df = pd.read_csv("/workspace/test/data_case_100.csv", encoding="utf-8-sig")

    if "text" not in df.columns or "answers" not in df.columns:
        raise ValueError("CSV ต้องมี column 'text' และ 'answers'")

    df['predicted_law'] = ""

    for i in range(df.shape[0]):
        context = df['text'].iloc[i]

        full_prompt = f"""
        ข้อความคดีนี้มีมาตราแพ่งและพาณิชย์ที่เกี่ยวข้องมากที่สุด 5 อันดับแรก มีอะไรบ้าง 
        """.strip()

        ans = ask_law_question(full_prompt, context)
        df.at[i, 'predicted_law'] = ans
        print(f"[{i}] ✅ Answer: {ans}")

    df_eval = df
    recall_k, mrr_k = evaluate_with_pytrec(df_eval, k=5)

    print(f"\n🎯 Recall@5 (เฉลี่ยจาก {len(df_eval)} แถว): {recall_k:.4f}")
    print(f"📈 MRR@5 (เฉลี่ยจาก {len(df_eval)} แถว): {mrr_k:.4f}")

    df_eval.to_csv("/workspace/test/output_predicted.csv", index=False, encoding="utf-8-sig")
