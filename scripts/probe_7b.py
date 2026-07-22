"""Does the bigger base (Qwen2.5-7B-Instruct) chat generally AND shop?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2

MODEL = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
print(f"loaded {MODEL}  |  VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

def ask(sysmsg, user, n=80):
    msgs = [{"role":"system","content":sysmsg},{"role":"user","content":user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt")["input_ids"].to("cuda")
    out = m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

print("\n--- GENERAL CHAT (plain assistant) ---")
for q in ["What's the capital of France?", "Tell me a short joke.",
          "How are you today?"]:
    print(f"\nQ: {q}\nA: {ask('You are a helpful assistant.', q)}")

print("\n--- SHOPPING (same model, shopping tool prompt) ---")
for q in ["I need a laptop.", "Something under $900 with 16GB RAM, lightweight."]:
    print(f"\nUser: {q}\nAgent: {ask(SYSTEM_PROMPT_V2, q)}")
