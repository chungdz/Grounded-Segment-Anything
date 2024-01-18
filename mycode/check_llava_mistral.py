from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoConfig, LlamaConfig 
import requests

class LlavaConfig(LlamaConfig):
    model_type = "llava"
AutoConfig.register("llava_mistral", LlavaConfig)

model = AutoModelForCausalLM.from_pretrained("openaccess-ai-collective/mistral-7b-llava-1_5-pretrained-projector")
processor = AutoProcessor.from_pretrained("openaccess-ai-collective/mistral-7b-llava-1_5-pretrained-projectorf")


image = Image.open("/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/001YG.mp4/000089.png")
prompt = "<image>\nFind important positional relations among only these objects: box,pillow,blanket,table,person,bed,bottle,clothes,phone,sofa,television,chair,laptop\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(0)
generate_ids = model.generate(**inputs, max_length=512, temperature=0.1, top_p=0.7, do_sample=True)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_token0104zation_spaces=False)[0])