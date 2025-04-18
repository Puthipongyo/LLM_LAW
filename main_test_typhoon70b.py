import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder

# ตั้งค่าป้องกัน CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 1. Load model & tokenizer
model_name = "scb10x/llama3.1-typhoon2-70b"
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

    print("📌 คำถาม:", question)
    print("📌 คำตอบจากโมเดล:\n", answer_only)
    print("=" * 80)

    if return_text:
        return answer_only

# 3. Main process
if __name__ == "__main__":
    df = pd.read_csv("/workspace/test/Extracted_Law_CSV.csv", encoding="utf-8-sig")

    if "text" not in df.columns:
        raise ValueError("ไม่พบ column 'text' ในไฟล์ CSV")

    # รวม context จาก 3 แถวแรก
    context = ""
    for i, row in df.head(3).iterrows():
        context += f"[{i}] {row['text']}\n\n"

    # ถามคำถาม
    # ask_law_question("จากเนื้อเรื่องนี้ มีมาตราไหนเกี่ยวข้องบ้าง", context)
    ask_law_question("""
    พระราชบัญญัติประกอบรัฐธรรมนูญว่าด้วยการเลือกตั ้งสมาชิกสภาผู ้แทนราษฎร  พ.ศ.  ๒๕๖๑  
มาตรา  ๑๖๙  เป็นบทบัญญัติที ่ให้สิทธิผู ้สมัครหรือพรรคการเมืองที ่ส่งสมาชิกของตนลงสมัครรับเลือกตั้ง 
ในเขตเลือกตั ้งนั ้นเป็นผู ้เสียหายตามประมวลกฎหมายวิธีพิจารณาความอาญา  เนื ่องจากการหาเสียงเลือกตั้ง  
ผู ้สมัครหรือพรรคการเมืองต้องเสียค่าใช้จ่ายในการเลือกตั ้ง  หากมีการกระทำความผิดตามกฎหมายเกี ่ยวกับ
 การเลือกตั ้งก่อให้เกิดความได้เปรียบเสียเปรียบกันระหว่างผู ้สมัคร  ผู ้สมัครหรือพรรคการเมืองย่อมได้รับ 
ความเสียหายโดยตรง  ส่วนการดำเนินคดีเลือกตั ้งตามรัฐธรรมนูญ  มาตรา  ๒๒๕  และมาตรา  ๒๒๖  วรรคเจ็ด   
มุ ่งหมายเพื ่อให้การจัดการเลือกตั ้งเป็นไปโดยสุจริตและเที ่ยงธรรม  ให้เสร็จสิ ้นไปโดยเร็ว  จึงบัญญัติ
 กระบวนการเฉพาะโดยให้หน้าที ่และอำนาจคณะกรรมการการเลือกตั ้งดำเนินคดีเลือกตั ้งในฐานะโจทก์ 
ต่อศาลฎีกาได้โดยตรง  และให้ใช้ระบบไต่สวนในการพิจารณา  การดำเนินคดีอาญาตามพระราชบัญญัติ 
ประกอบรัฐธรรมนูญว่าด้วยการเลือกตั ้งสมาชิกสภาผู ้แทนราษฎร  พ.ศ.  ๒๕๖๑  มาตรา  ๑๖๙  และ 
การดำเนินคดีเลือกตั ้งตามรัฐธรรมนูญจึงมีวัตถุประสงค์และเจตนารมณ์แตกต่างกัน  การกำหนดผู ้มีอำนาจ 
ฟ้องคดี  เขตอำนาจศาล  หรือระบบในการพิจารณาคดีย่อมแตกต่างกันตามลักษณะของคดี  บทบัญญัติ 
มาตรา  ๑๖๙  ไม่ขัดหรือแย้งต่อรัฐธรรมนูญ  มาตรา  ๑๙๔  วรรคหนึ ่ง  มาตรา  ๒๒๕  และมาตรา  ๒๒๖  
วรรคเจ็ด
ข้อความนี้มีมาตราอะไรเกี่ยวข้อง
    """, context)
