from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoProcessor
from huggingface_hub import login
import json
import re
import torch

login("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")

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
# tokenizer.pad_token = tokenizer.eos_token
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", device_map='auto')

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
# tokenizer.pad_token = tokenizer.eos_token
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map='auto')
# print(model.hf_device_map)

# tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
# tokenizer.pad_token = tokenizer.eos_token
# model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5", device_map='auto')
# print(model.hf_device_map)

# tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-33b-v1.3")
# tokenizer.pad_token = tokenizer.eos_token
# model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-33b-v1.3", device_map='auto')
# print(model.hf_device_map)


# tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat")
# model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B-Chat", torch_dtype=torch.float16, device_map='auto')
# print(model.hf_device_map)

tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardLM-13B-V1.2")
model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardLM-13B-V1.2", device_map='auto')
# model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardLM-13B-V1.2", device_map='auto')

print(model.hf_device_map)

with open("/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/rel.json", 'r') as f:
    wrong_prompt = f.readlines()

wdicts = [json.loads(line) for line in wrong_prompt]

rel_list = "[on,behind,in_front_of,on_the_side_of,above,beneath,drinking_from,have_it_on_the_back,wearing,holding,lying_on,covered_by,carrying,eating,leaning_on,sitting_on,twisting,writing_on,standing_on,touching,wiping,at,under,near]"
rel_list_0 = "[on,behind,in_front_of,on_the_side_of,above,beneath,at,under,near]"
rel_list_1 = "[drinking_from,have_it_on_the_back,wearing,holding,lying_on,covered_by,carrying,eating,leaning_on,sitting_on,twisting,writing_on,standing_on,touching,wiping]"

desc_text = wdicts[10]['llava']
desc_index = desc_text.index(("ASSISTANT"))
desc = desc_text[desc_index + 11:]
obstr = "window, person, blanket, glass, box, closet, bottle, sofa, hands, floor, cup"

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
# obstr = "pillow,door,cup,bottle,shoe,hands,clothes,person,floor,box,phone,table,bed,closet"
# desc = "The person is standing on a red floor, wearing a blue shirt and black pants. They are holding a pair of black shoes in their hands. The floor is next to a bed, and there is a cup and a bottle nearby. A door is also present in the scene."

# obstr = "laptop,broom,mirror,blanket,cup,bottle,shoe,person,floor,picture,box,bag,couch,cabinet"
# desc = "The person is cleaning the floor with a broom. The floor is white, and the person is standing on it. There is a couch in the room, and a cup is placed on a surface. A bottle is also present in the scene. \
# The person is wearing a shoe, and there is a mirror in the room. A picture is hanging on the wall, and a box is located nearby. A bag is placed on the floor, and a chair is positioned in the room."

obstr = "sofa,blanket,cup,glass,bottle,hands,window,floor,person,box,closet"
desc = "The woman is holding a cup and a bottle. The cup is placed on the floor, and the bottle is located near the cup. The woman is standing in a bathroom, and there is a window nearby. The floor is covered with a rug, and a toilet is present in the bathroom. The woman is wearing a pink shirt, and ther \
e is a pink towel in the bathroom. A sofa is also visible in the scene, along with a blanket."

# obstr = "pillow,laptop,cup,television,bottle,book,hands,person,bag,floor,table"
# desc = "The person is sitting on a bed with a laptop on a table in front of them. There is a cup on the table, and a bottle is placed nearby. A book is also present on the table. The person is holding a bag, and there is a television in the room. The floor is covered with a carpet."

prompt = '''Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].
Requirements:

1. Only detect relations in this list: [on,behind,in_front_of,on_the_side_of,above,beneath,drinking_from,have_it_on_the_back,wearing,holding,lying_on,covered_by,carrying,eating,leaning_on,sitting_on,twisting,writing_on,standing_on,touching,wiping,at,under,near].
2. Only detect subjects and objects in this list: [{}].
3. The output should contain at most 15 tuples ranked by their importance. Person is more important.

For example:
Input text: "In the image, the woman sits on the bed near a pillow. There is a cup on the table, and a bottle is placed nearby. A book is also present on the table.". 
Answer: [["woman", "sitting_on", "bed"], 
            ["woman", "near", "pillow"],
            ["cup", "on", "table"],
            ["bottle", "near", "table"],
            ["book", "on", "table"]]

Input text:{}
Answer:'''.format(obstr, desc)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
# generation_output = model.generate(input_ids=input_ids, max_length=2048, do_sample=False, temperature=None, top_p=None)
generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)

res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
answer_index = [m.start() for m in re.finditer("Answer:", res)][-1]
answer = res[answer_index:]

struct_start = answer.index('[[')
struct_end = answer.index(']]')
rels = json.loads(answer[struct_start: struct_end + 2])

print(rels)

prompt = '''Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].
Requirements:

1. Only detect relations in this list: {}, if the relation is not in the array, then discard the tuple or try to find relation with similar meaning.
2. Only detect subjects and objects in the given SO list. If the subject or object is not in the array, then discard the tuple.
3. The output should contain at most 15 tuples ranked by their importance. Person is more important.
4. Do not infer or assume relations. Only depend on sentences themself.
5. Generate answers, thinking, correctness step by step. Correct the wrong thing of previous step.

For example:

SO list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]
Input text: "In the image, the woman sits on the bed near by a pillow. The women's hands is holding the bag. The woman is wearing a blue shirt and white shoes. There is a cup on the table, and a bottle is placed nearby. 
            A book and a monitor is also present on the table. A chair is in the room and in the front of the table". 

Step 1 Answer: [["woman", "sitting", "bed"], 
            ["woman", "near_by", "pillow"],
            ["woman", "is_holding", "bag"],
            ["woman", "wearing", "shirt"],
            ["wearing", "shoes"],
            ["cup", "on", "table"],
            ["bottle", "nearby"],
            ["book", "on", "table"],
            ["monitor", "on", "table"],
            ["chair", "in_the_front_of", "table"],
            ["chair", "in_room"]]
Thinking: Check the step 1 answer and think about the relation tuple format. Does the tuples all contain three elements? Does relations separate from subjects and objects? And generate next step answer.
Correctness: ["wearing", "shoes"] has no subject. It should be changed into ["woman", "wearing", "shoes"]
            ["bottle", "near_by"] has no object. It should be changed into ["bottle", "near_by", "cup"]. 
            ["chair", "in_room"] only has two elements, and "in" should be relation, "room" should be object. It should be changed into ["chair", "in", "room"].

Step 2 Answer: [["woman", "sitting", "bed"], 
            ["woman", "near_by", "pillow"],
            ["woman", "is_holding", "bag"],
            ["woman", "wearing", "shirt"],
            ["woman", "wearing", "shoes"],
            ["cup", "on", "table"],
            ["bottle", "nearby", "cup"],
            ["book", "on", "table"],
            ["monitor", "on", "table"],
            ["chair", "in_the_front_of", "table"],
            ["chair", "in", "room"]]
Thinking: Check the step 2 answer and think about the relation names. Does all relations in the tuples are in the relation list in requirement 1? Can I replace wrong one with correct one? And generate next step answer.
Correctness: "Sitting" in ["woman", "sitting", "bed"] should be replaced by "sitting_on". 
            "near_by" in ["woman", "near_by", "pillow"]  should be replaced by "near". 
            "is_holding" in ["woman", "is_holding", "bag"] should be replaced by "holding". 
            "nearby" in ["bottle", "nearby", "cup"]  should be replaced by "near".
            "in_the_front_of" in ["chair", "in_the_front_of", "table"] should be replaced by "in_front_of". 

Step 3 Answer: [["woman", "sitting_on", "bed"], 
            ["woman", "near", "pillow"],
            ["woman", "holding", "bag"],
            ["woman", "wearing", "shirt"],
            ["woman", "wearing", "shoes"],
            ["cup", "on", "table"],
            ["bottle", "near", "cup"],
            ["book", "on", "table"],
            ["monitor", "on", "table"],
            ["chair", "in_front_of", "table"],
            ["chair", "in", "room"]]
Thinking: Check the step 3 answer and think about the relation names. Does subjects and objects in the tuples are in the SO list? Can I replace wrong one with correct one? And generate final answer.
Correctness: "woman" should be replaced by "person".
            "shirt" is not in the SO list. It should be removed.
            "room" is not in the SO list. It should be removed.
            "monitor" should be replaced by "screen".

Final Answer: [["person", "sitting_on", "bed"],
            ["person", "near", "pillow"],
            ["person", "holding", "bag"],
            ["person", "wearing", "shoes"],
            ["cup", "on", "table"],
            ["bottle", "near", "cup"],
            ["book", "on", "table"],
            ["screen", "on", "table"],
            ["chair", "in_front_of", "table"]]

Now based on the example and requirements, generate answers, thinking, correctness step by step.

SO list: [{}]
Input text:{}
Step 1 Answer:'''.format(rel_list, obstr, desc)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
# generation_output = model.generate(input_ids=input_ids, max_length=2048, do_sample=False, temperature=None, top_p=None)
generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)

res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
answer_index = [m.start() for m in re.finditer("Answer:", res)][-1]
answer = res[answer_index:]

struct_start = answer.index('[[')
struct_end = answer.index(']]')
rels = json.loads(answer[struct_start: struct_end + 2])

print(rels)

# no reasoning

prompt = '''Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].
Requirements:

1. Only detect relations in this list: {}, if the relation is not in the array, then discard the tuple or try to find relation with similar meaning.
2. Only detect subjects and objects in the given SO list. If the subject or object is not in the array, then discard the tuple.
3. The output should contain at most 15 tuples ranked by their importance. Person is more important.
4. Do not infer or assume relations. Only depend on sentences themself.

For example:

SO list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]
Input text: "In the image, the woman sits on the bed near by a pillow. The women's hands is holding the bag. The woman is wearing a blue shirt and white shoes. There is a cup on the table, and a bottle is placed nearby. 
            A book and a monitor is also present on the table. A chair is in the room and in the front of the table". 

Answer: [["person", "sitting_on", "bed"],
        ["person", "near", "pillow"],
        ["person", "holding", "bag"],
        ["person", "wearing", "shoes"],
        ["cup", "on", "table"],
        ["bottle", "near", "cup"],
        ["book", "on", "table"],
        ["screen", "on", "table"],
        ["chair", "in_front_of", "table"]]

Now based on the example and requirements, generate the answer.

SO list: [{}]
Input text:{}
Answer:'''.format(rel_list, obstr, desc)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
# generation_output = model.generate(input_ids=input_ids, max_length=2048, do_sample=False, temperature=None, top_p=None)
generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)

res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
answer_index = [m.start() for m in re.finditer("Answer:", res)][-1]
answer = res[answer_index:]

struct_start = answer.index('[[')
struct_end = answer.index(']]')
rels = json.loads(answer[struct_start: struct_end + 2])

print(rels)

# one step reasoning

prompt = '''Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].
Requirements:

1. Only detect relations in this list: {}, if the relation is not in the array, then discard the tuple or try to find relation with similar meaning.
2. Only detect subjects and objects in the given SO list. If the subject or object is not in the array, then discard the tuple.
3. The output should contain at most 15 tuples ranked by their importance. Person is more important.
4. Do not infer or assume relations. Only depend on sentences themself.
5. Generate answers, thinking, correctness step by step. Correct the wrong thing of previous step.

For example:

SO list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]
Input text: "In the image, the woman sits on the bed near by a pillow. The women's hands is holding the bag. The woman is wearing a blue shirt and white shoes. There is a cup on the table, and a bottle is placed nearby. 
            A book and a monitor is also present on the table. A chair is in the room and in the front of the table". 

Step 1 Answer: [["person", "sitting_on", "bed"], 
            ["person", "near", "pillow"],
            ["person", "holding", "bag"],
            ["wearing", "shoes"],
            ["cup", "on_table"],
            ["bottle", "near"],
            ["book", "on", "table"],
            ["screen", "on", "table"],
            ["chair", "in_front_of", "table"],
            
Thinking: Check the step 1 answer and think about the relation tuple format. Does the tuples all contain three elements? Does relations separate from subjects and objects? And generate next step answer.
Correctness: ["wearing", "shoes"] has no subject. It should be changed into ["woman", "wearing", "shoes"]
            ["cpu", "on_room"] only has two elements, and "on" should be relation, "table" should be object. It should be changed into ["cup", "on", "table"].
            ["bottle", "near"] has no object. It should be changed into ["bottle", "near", "cup"]. 
            

Final Answer: [["person", "sitting_on", "bed"],
            ["person", "near", "pillow"],
            ["person", "holding", "bag"],
            ["person", "wearing", "shoes"],
            ["cup", "on", "table"],
            ["bottle", "near", "cup"],
            ["book", "on", "table"],
            ["screen", "on", "table"],
            ["chair", "in_front_of", "table"]]

Now based on the example and requirements, generate answers, thinking, correctness step by step.

SO list: [{}]
Input text:{}
Step 1 Answer:'''.format(rel_list, obstr, desc)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
# generation_output = model.generate(input_ids=input_ids, max_length=2048, do_sample=False, temperature=None, top_p=None)
generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)

res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
answer_index = [m.start() for m in re.finditer("Answer:", res)][-1]
answer = res[answer_index:]

struct_start = answer.index('[[')
struct_end = answer.index(']]')
rels = json.loads(answer[struct_start: struct_end + 2])

print(rels)

# one step reasoning

prompt = '''Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].
Requirements:

1. Only detect relations in RE list. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning. RE list: {}.
2. Only detect subjects and objects in the given SO list. If the subject or object is not in the array, then discard the tuple.
3. The output should contain at most 8 tuples ranked by their importance. Person is more important.
4. Do not infer or assume relations. Only depend on sentences themself.

For example:

SO list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]
Input text: "In the image, the woman sits on the bed near by a pillow. The women's hands is holding the bag. The woman is wearing a blue shirt and white shoes. There is a cup on the table, and a bottle is placed nearby. 
            A book and a monitor is also present on the table. A chair is in the room and in the front of the table". 

Thinking 1: 
        Guideline: All should contain three elements. Relations should separate from subjects and objects. Therefore based on the requirement and input text:
            1.["cup", "on_the_table"] is wrong and should be changed into ["cup", "on", "table"].
            2.["wearing", "shoes"] has no subject. It should be changed into ["woman", "wearing", "shoes"].
            3.["bottle", "near"] has no object. It should be changed into ["bottle", "near", "cup"].
            4.["person", "holding", "bag", "person"] is wrong as it is larger than three elements in tuple and should be change into ["person", "holding", "bag"].
Thinking 2: 
        Guideline: Here is RE list: {}. It needs to be double checked to make sure all relations in the mid of the tuples should be exact the same as ones in the RE list. Therefore based on the requirement and input text:
            1."sitting" in ["person", "sitting", "bed"] should be replaced by "sitting_on". 
            2."near_by" in ["person", "near_by", "pillow"] should be replaced by "near". 
            3."is_holding" in ["person", "is_holding", "bag"] should be replaced by "holding". 
            5."in_the_front_of" in ["chair", "in_the_front_of", "table"] should be replaced by "in_front_of". 
            6. "on_the" in ["screen", "on_the", "table"] should be replaced by "on".
Thinking 3: 
        Guideline: Here is SO list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]. Subjects and object at edge of the tuple should be exact the same as ones in the SO list, therefore based on the requirement and input text:
            1."women" is not in SO list, so "woman" should be replaced by "person". 
            2."shirt" is not in the SO list. It should be removed.
            3."room" is not in the SO list. It should be removed.
            4."monitor" is not in SO list, so it should be replaced by "screen".
            5. "shoes" is not in SO list, it should be replaced by "shoe".
Thinking 4: 
        Guideline: all relations should make sense, and person is more important. Therefore based on the requirement and input text:
            1. ["table", "on", "cup"] is not making sense. It should be changed into ["cup", "on", "table"].
            2. ["hands", "holding", "bag"] is correct. But person is more important, so it should be changed into ["person", "holding", "bag"].
            3. ["bag", "holding", "person"] has wrong order. It should be changed into ["person", "holding", "bag"].
Thinking 5: 
        Guideline: find the correct tuples based on above thinkings and generate the final answer.
                
Answer: [["person", "sitting_on", "bed"],
        ["person", "near", "pillow"],
        ["person", "holding", "bag"],
        ["person", "wearing", "shoe"],
        ["cup", "on", "table"],
        ["bottle", "near", "cup"],
        ["book", "on", "table"],
        ["screen", "on", "table"],
        ["chair", "in_front_of", "table"]]
END

Now based on the example and requirements, generate the same steps of thinkings as the example by repeating the guidelines and then give reasoning process. Do not over thinking too much. Then generate answer and finish with END.

SO list: [{}]
Input text:{}

Thinking 1:'''.format(rel_list, rel_list, obstr, desc)

prompt = '''Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].
Requirements:

1. Only detect relations in RE list. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning. RE list: {}.
2. Only detect subjects and objects in the given SO list. If the subject or object is not in the array, then discard the tuple.
3. The output should contain at most 8 tuples ranked by their importance. Person is more important.
4. Do not infer or assume relations. Only depend on sentences themself.

For example:

SO list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]
Input text: "In the image, the woman sits on the bed near by a pillow. The women's hands is holding the bag. The woman is wearing a blue shirt and white shoes. There is a cup on the table, and a bottle is placed nearby. 
            A book and a monitor is also present on the table. A chair is in the room and in the front of the table". 

Thinking 1: 
        Guideline: All should contain three elements. Relations should separate from subjects and objects. Therefore based on the requirement and input text:
Thinking 2: 
        Guideline: Here is RE list: {}. It needs to be double checked to make sure all relations in the mid of the tuples should be exact the same as ones in the RE list. Therefore based on the requirement and input text:
Thinking 3: 
        Guideline: Here is SO list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]. Subjects and object at edge of the tuple should be exact the same as ones in the SO list, therefore based on the requirement and input text:
Thinking 4: 
        Guideline: all relations should make sense, and person is more important. Therefore based on the requirement and input text:
Thinking 5: 
        Guideline: find the correct tuples based on above thinkings and generate the final answer.
                
Answer: [["person", "sitting_on", "bed"],
        ["person", "near", "pillow"],
        ["person", "holding", "bag"],
        ["person", "wearing", "shoe"],
        ["cup", "on", "table"],
        ["bottle", "near", "cup"],
        ["book", "on", "table"],
        ["screen", "on", "table"],
        ["chair", "in_front_of", "table"]]
END

Now based on the example and requirements, generate the same steps of thinkings as the example by repeating the guidelines and then give reasoning process. Do not over thinking too much. Then generate answer and finish with END.

SO list: [{}]
Input text:{}

Thinking 1:'''.format(rel_list, rel_list, obstr, desc)

# messages = [
#     {"role": "user", "content": prompt}
# ]
# input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# output_ids = model.generate(input_ids.to('cuda'))
# res = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
# generation_output = model.generate(input_ids=input_ids, max_length=2048, do_sample=False, temperature=None, top_p=None)
generation_output = model.generate(input_ids=input_ids, max_length=4096, temperature=0.1, top_p=0.7, do_sample=True)

res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
answer_index = [m.start() for m in re.finditer("Answer:", res)][-1]
answer = res[answer_index:]

struct_start = answer.index('[[')
struct_end = answer.index(']]')
rels = json.loads(answer[struct_start: struct_end + 2])

print(rels)


