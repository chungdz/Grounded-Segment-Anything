from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import requests

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", device_map="auto")
print(model.hf_device_map)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")

# image = Image.open("/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/001YG.mp4/000089.png")
# prompt = "<image>\nFind important positional relations among only these objects: box,pillow,blanket,table,person,bed,bottle,clothes,phone,sofa,television,chair,laptop\nASSISTANT:"
prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
image = Image.open(requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(0)
generate_ids = model.generate(**inputs, max_length=512, temperature=0.1, top_p=0.7, do_sample=True)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_token0104zation_spaces=False)[0])
