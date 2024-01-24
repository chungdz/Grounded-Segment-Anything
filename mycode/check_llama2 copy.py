from transformers import LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import login
import json
import re

login("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")

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map='auto')
print(model.hf_device_map)

with open("/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/rel.json", 'r') as f:
    wrong_prompt = f.readlines()

wdicts = [json.loads(line) for line in wrong_prompt]

desc_text = wdicts[0]['llava']
desc_index = desc_text.index(("ASSISTANT"))
desc = desc_text[desc_index + 11:]

# p1 = '''
# You as the assistant please generate tuples for me [Entity, Relation, Entity] from quoted text which is describing the image. The tuples show most important relations between two entities in the sentences. 

# Requirements:
# 1. The relation are limited in following array: [on, behind, in_front_of, on_the_side_of, above, beneath, drinking_from, have_it_on_the_back, wearing, holding, lying_on, covered_by, carrying, eating, leaning_on, sitting_on, twisting, writing_on, standing_on, touching, wiping, at, under, near] You should only use relations I give you. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning.
# 2. Do not infer or assume relations. Only depend on sentences themself.
# 3. Use a valid Json format to form the output. Start and end with curly brackets.
# 4. The relation should make sense.
# 5. The output should contain at most 15 tuples

# For example:

# Input test: "In the image, the woman sits on the bed near a pillow".
# Output: {[["woman", "on", "bed"],
#         ["woman", "near", "pillow"]]}

# Now, You as the assistant generate tuples of relation for me with the following quoted input text:
# '''
# prompt = p1 + '"{}"\nASSISTANT:'.format(desc)

p1 = '''Generate tuples for me [Entity, Relation, Entity] from quoted text which is describing the image. The tuples show most important relations between two entities in the sentences.                                                                                                                                   
                                                                                                                                                                                                                                                                                                                    
Requirements:                                                                                                                                                                                                                                                                                                       
1. The RELATION are ONLY LIMITED in following array: [on, behind, in_front_of, on_the_side_of, above, beneath, drinking_from, have_it_on_the_back, wearing, holding, lying_on, covered_by, carrying, eating, leaning_on, sitting_on, twisting, writing_on, standing_on, touching, wiping, at, under, near] You shoul
d only use RELATION I give you. If the RELATION is not in the array, then discard the tuple or try to find RELATION with similar meaning.                                                                                                                                                                           
2. Do not infer or assume relations. Only depend on sentences themself.                                                                                                                                                                                                                                             
3. Use a valid Json format to form the output. Start and end with curly brackets.                                                                                                                                                                                                                                   
4. The relation should make sense.                                                                                                                                                                                                                                                                                  
5. The output should contain at most 15 tuples ranked by their importance.                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                    
For example:                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                    
Input text: "In the image, the woman sits on the bed near a pillow".                                                                                                                                                                                                                                                
Assistant output tuples: {[["woman", "sitting_on", "bed"],                                                                                                                                                                                                                                                          
        ["woman", "near", "pillow"]]}                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                    
Input text: '''
prompt = p1 + '"{}"\nOutput tuples:'.format(desc)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.05, do_sample=False)
# generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)

res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
answer_index = [m.start() for m in re.finditer("Assistant output tuples:", res)][-1]
answer = res[answer_index:]

p2 = '''You as the assistant read carefully about the examples and requirements listed below, and generate tuples for me [ENTITY, RELATION, ENTITY] from following quoted text which is describing the image.

Requirements:
1. The RELATION are ONLY LIMITED in following array: [on, behind, in_front_of, on_the_side_of, above, beneath, drinking_from, have_it_on_the_back, wearing, holding, lying_on, covered_by, carrying, eating, leaning_on, sitting_on, twisting, writing_on, standing_on, touching, wiping, at, under, near] You should only use relations I give you. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning.
2. Do not infer or assume relations. Only depend on sentences themself.
3. Use a valid Json format to form the output. Start and end with curly brackets.
4. The relation should make sense.
5. The output should contain at most 15 tuples

For example:

Input text: "In the image, the woman sits on the bed near a pillow".
Assistant output tuples: {[["woman", "sitting_on", "bed"],
        ["woman", "near", "pillow"]]}

The following is your previous answer, now check it based on the requirements and examples above. Do not use any RELATION that is not in the RELATION array.

'''

prompt2 = p2 + \
            'Input text: "{}"\n{}'.format(desc, answer) + \
            '\nGenerate new one if you find anything wrong, or keep the same one if you think it is correct.\n' + 'Input text: "{}"\n Assistant output tuples:'.format(desc)

print(prompt2)

input_ids = tokenizer(prompt2, return_tensors="pt", padding=True).input_ids.to(0)
generation_output2 = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)

res2 = tokenizer.batch_decode(generation_output2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res2)

struct_start = answer.index('{')
struct_end = answer.index('}')
rels = json.loads(answer[struct_start + 1: struct_end])

print(rels)
