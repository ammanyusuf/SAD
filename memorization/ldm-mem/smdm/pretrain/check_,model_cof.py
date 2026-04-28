from transformers import AutoModelForCausalLM
cfg = AutoModelForCausalLM.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
