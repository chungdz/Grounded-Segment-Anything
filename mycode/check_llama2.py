from transformers import LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import login
import json

login("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")

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')
print(model.hf_device_map)

prompt = '''
Generate tuples [Entity, Relation, Entity] from quoted text which is describing the image. The tuples show most important relations between two entities in the sentences. 

Requirements:
1. The relation are limited in following array: [on, behind, in_front_of, on_the_side_of, above, beneath, drinking_from, have_it_on_the_back, wearing, holding, lying_on, covered_by, carrying, eating, leaning_on, sitting_on, twisting, writing_on, standing_on, touching, wiping, at, under, near] You should only use relations I give you. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning.
2. Do not infer or assume relations. Only depend on sentences themself.
3. Use a valid Json format to form the output. Start and end with curly brackets.
4. The relation should make sense.
5. The output should contain at most 15 tuples

For example:

Input test: "In the image, the woman sits on the bed near a pillow".
Output: {[["woman", "on", "bed"],
        ["woman", "near", "pillow"]]}

Now, generate tuples of relation for me with this input text:

"In the image, the person is sitting on a chair in front of a table, which has a laptop on it. The table is located next to a bed, and there is a bottle on the table as well. A sofa is also present in the room, and a television is positioned nearby. A phone is placed on the table, and a blanket and clothes are located on the bed. The person is wearing a blue shirt, and there is a computer mouse on the table."
\nASSISTANT:'''
input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)

res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
answer_index = res.index("ASSISTANT")
answer = res[answer_index + 11:]

struct_start = answer.index('{')
struct_end = answer.index('}')
struct_str = answer[struct_start + 1: struct_end].strip('\n').strip(' ')