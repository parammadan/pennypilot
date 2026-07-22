"""Probe: can the BASE Qwen chat generally, and does the RL-trained one still?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_Q = ["What's the capital of France?",
         "Tell me a short joke.",
         "Explain photosynthesis in one sentence."]

def ask(model, tok, sysmsg, user):
    msgs=[{"role":"system","content":sysmsg},{"role":"user","content":user}]
    ids=tok.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,
                                return_dict=True,return_tensors="pt")["input_ids"].to("cuda")
    out=model.generate(ids,max_new_tokens=60,do_sample=False,pad_token_id=tok.pad_token_id)
    return tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()

tok=AutoTokenizer.from_pretrained(MODEL)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to("cuda").eval()
print("="*60,"\nBASE Qwen2.5-1.5B-Instruct (NO shopping training), plain assistant:")
for q in GEN_Q:
    print(f"\nQ: {q}\nA: {ask(base, tok, 'You are a helpful assistant.', q)}")

print("\n"+"="*60,"\nRL-TRAINED shopping policy, SAME general questions (shopping sys prompt):")
from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
trained=PeftModel.from_pretrained(base,"/scratch/madan.pa/pennypilot/rloo50_v2/policy").eval()
for q in GEN_Q[:2]:
    print(f"\nQ: {q}\nA: {ask(trained, tok, SYSTEM_PROMPT_V2, q)}")
