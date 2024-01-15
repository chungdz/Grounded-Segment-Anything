from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_13b")
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_13b", device_map='auto')
print(model.hf_device_map)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda(0)
generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)

print(generation_output)
