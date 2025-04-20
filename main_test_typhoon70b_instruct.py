import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder
import pytrec_eval


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


model_name = "scb10x/llama3.1-typhoon2-70b-instruct"
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
        บริบท:\n{context}\n\nคำถาม:\n{question}
        """},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.7,
        top_p=0.95,
    )
    response = outputs[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(response, skip_special_tokens=True)

    return decoded.strip()

# ฟังก์ชันแยกเฉพาะเลขมาตรา เช่น 1336
def extract_law_numbers(text):
    return re.findall(r'\d+', text)

def evaluate_with_pytrec(df, k=5):
    qrel = {}  # ground truth
    run = {}   # predicted

    for idx, row in df.iterrows():
        qid = f"q{idx}"
        true_ids = extract_law_numbers(str(row['answers']))
        pred_ids = extract_law_numbers(str(row['predicted_law']))[:k]

        qrel[qid] = {law_id: 1 for law_id in true_ids}

        run[qid] = {law_id: 1.0 / (i + 1) for i, law_id in enumerate(pred_ids)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall'})
    results = evaluator.evaluate(run)


    per_row_recalls = []
    for qid, metrics in results.items():
        row_idx = int(qid[1:])
        recall_score = metrics.get(f'recall_{k}', 0.0)
        df.at[row_idx, 'recall_at_k'] = recall_score
        per_row_recalls.append(recall_score)

    average_recall = sum(per_row_recalls) / len(per_row_recalls) if per_row_recalls else 0.0
    return average_recall

# Main process
if __name__ == "__main__":
    df = pd.read_csv("/workspace/test/data_case_100.csv", encoding="utf-8-sig")

    if "text" not in df.columns or "answers" not in df.columns:
        raise ValueError("CSV ต้องมี column 'text' และ 'answers'")

    df['predicted_law'] = ""

    for i in range(df.shape[0]):
        context = df['text'].iloc[i]
        # context = f"{question}"

        full_prompt = f"""
        ข้อความคดีนี้มีมาตราแพ่งและพาณิชย์ที่เกี่ยวข้องมากที่สุด 5 อันดับแรก มีอะไรบ้าง 
        """.strip()

        ans = ask_law_question(full_prompt, context)
        df.at[i, 'predicted_law'] = ans
        print(f"[{i}] ✅ Answer: {ans}")

        if i == 50:  # ทดสอบแค่ 3 แถวแรก
            break

    # ประเมิน Recall@5
    df_eval = df.head(50)
    recall_k = evaluate_with_pytrec(df_eval, k=5)


    print(f"\n🎯 Recall@5 (เฉลี่ยจาก {len(df_eval)} แถว): {recall_k:.4f}")

    # บันทึกผล
    df_eval.to_csv("/workspace/test/output_predicted.csv", index=False, encoding="utf-8-sig")
