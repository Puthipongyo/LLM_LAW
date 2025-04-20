import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder

# ตั้งค่าป้องกัน CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 1. Load model & tokenizer
model_name = "scb10x/llama3.1-typhoon2-70b-instruct"
token = HfFolder.get_token()  # ใช้ token ที่ login ไว้ในเครื่องนี้

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

# 2. ฟังก์ชันถามคำถามจาก context
def ask_law_question(question, context, return_text=False):
    prompt = f"""<|user|>
    ข้อมูลดังนี้:

    \"\"\"{context}\"\"\"

    คำถาม: {question}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=600, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ตัด prompt ออก ให้เหลือแค่คำตอบ
    answer_only = response.split("คำถาม:")[-1].strip()

    print("📌 คำตอบจากโมเดล:\n", answer_only)
    print("=" * 80)

    if return_text:
        return answer_only

# 3. Main process
if __name__ == "__main__":
    df = pd.read_csv("/workspace/test/data_case_100.csv", encoding="utf-8-sig")

    if "text" not in df.columns:
        raise ValueError("ไม่พบ column 'text' ในไฟล์ CSV")

    # รวม context จาก 3 แถวแรก
    context = ""
    for i, row in df.head(3).iterrows():
        context += f"[{i}] {row['text']}\n\n"

    # ถามคำถาม
    # ask_law_question("จากเนื้อเรื่องนี้ มีมาตราไหนเกี่ยวข้องบ้าง", context)
    ask_law_question("""
        ที่ดินโฉนดที่ 499 เนื้อที่ 56 ไร่ 68 ตารางวาเดิมเป็นกรรมสิทธิ์ของนายซอบิดาซึ่งถึงแก่กรรมโดยมิได้ทำพินัยกรรมโจทก์มีสิทธิได้รับมรดก 1 ใน 13 เป็นเนื้อที่ 4 ไร่ 1 งาน 29 ตารางวา ขอให้พิพากษาให้โจทก์ได้รับส่วนมรดกดังกล่าวร่วมกับจำเลย
    จำเลยต่อสู้ว่า ทายาทได้ทำบันทึกแบ่งมรดกตามคำสั่งผู้ตายไว้ โจทก์มีส่วนได้เพียง 1 ไร่ 3 งาน 14 ตารางวาเศษ คดีโจทก์ขาดอายุความมรดก
    นางนิ่ม โซ๊ะสลาม ในฐานะส่วนตัวและมารดาผู้แทนโดยชอบธรรมของนางสาวแสงทอง เด็กหญิงซาฟียะ นางสาวอามีเนาะ นางทองดี นายสว่างร้องสอดขอรับส่วนแบ่งมรดกในฐานะภริยาและบุตรเจ้ามรดก
    ได้ความในเบื้องต้นว่า นายซอเจ้ามรดกมีภริยา 2 คน คือนางเชย และนางนิ่ม มีบุตรรวม 12 คน หลังเจ้ามรดกตาย บุตรทั้ง 12 คน ได้ลงนามทำบันทึกแบ่งมรดกตามเอกสาร ล.1 นางนิ่ม พิมพ์ลายนิ้วมือเป็นพยาน
    ข้อความคดีนี้มีมาตราแพ่งและพาณิชย์ที่เกี่ยวข้องอะไรบ้าง (บอกเเค่มาตรา)
    """, context)
