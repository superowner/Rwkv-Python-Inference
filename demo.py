import web_rwkv_py as wrp
import numpy as np

model = wrp.Model("E:/RWKV-Runner_windows_x64/models/RWKV-x060-World-3B-v2.1-Claude-nsfw.st")
tokenizer = wrp.Tokenizer("E:/RWKV-Runner_windows_x64/backend-rust/assets/rwkv_vocab_v20230424.json")

# prompt = "The Eiffel Tower is"
# tokens = tokenizer.encode(prompt)
# output, state = model.run(tokens)
# print(prompt, end="")

# We must explicitly call this to clone the state.
# state_1 = model.clone_state(state)

prompt = "现在请你扮演猫娘和我交流，并且要尊称我为主人，你回复的第一句话要问候我。"
tokens = tokenizer.encode(prompt)
print(prompt)
output, state_2 = model.run(tokens)
# output_token, _ = max(enumerate(output), key=lambda x: x[1])
cc=sorted(enumerate(output), key=lambda x: x[1])
cc=[(x[0]-1) for x in cc]
print('\ncc:\n',cc[0:10],'\n')
_output=np.array(output)
print('\noutput_token:',type(output),'output_token:',len(output),'list:',_output.shape)

dd=[]
counter=0
for x in cc:
    if x==0:
        break
    print('counter:',counter)
    dd.append(x)
    counter+=1

print(len(dd))
str = str(bytearray(tokenizer.decode(dd)), encoding='utf-8')
print(str)

