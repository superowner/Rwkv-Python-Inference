########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys
import numpy as np
import web_rwkv_py as wrp
import torch
from torch.nn import functional as F
from RWKV import RWKV
from rwkv_tokenizer import TRIE_TOKENIZER

class Chat:
    def __init__(
        self, 
        temperature=1.0,
        top_p=0.3,#0.85,
        top_k=0,
        alpha_frequency=1,#0.2,#频率惩罚
        alpha_presence=0,#0.2,#存在惩罚
        alpha_decay=0.996,#惩罚衰减
        token_ban=[],
        token_stop=[],
        chunk_len=256,#并行token块大小？
        ):
        #
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency  # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence  # Presence Penalty (as in GPT-3)
        self.alpha_decay = alpha_decay  # gradually decay the penalty
        self.token_ban = token_ban  # ban the generation of some tokens
        self.token_stop = token_stop  # stop generation whenever you see any token here
        self.chunk_len = (
            chunk_len  # split input into chunks to save VRAM (shorter -> slower)
        )

    def load(self,model_path_st:str,vocab_path_txt:str):
        rwkv = RWKV(model_path_st)
        # self.tokenizer = wrp.Tokenizer("E:/RWKV-Runner_windows_x64/backend-rust/assets/rwkv_vocab_v20230424.json")
        
        self.tokenizer = TRIE_TOKENIZER(vocab_path_txt)
        self.model = rwkv #wrp.Model("E:/RWKV-Runner_windows_x64/models/RWKV-x060-World-3B-v2.1-Claude-nsfw.st")
        print('model_version:',self.model.version)
        pass

    def encode(self, x:str):
        # print("t:",str(type(self.tokenizer)))
        if "Tokenizer" in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)
    
    def np_softmax(self, x: np.ndarray, axis: int):
        x -= x.max(axis=axis, keepdims=True)
        e: np.ndarray = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)
    
    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        if type(logits) == list:
            logits = np.array(logits)
        np_logits = type(logits) == np.ndarray
        if np_logits:
            probs = self.np_softmax(logits, axis=-1)
        else:
            probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        # 'privateuseone' is the type of custom devices like `torch_directml.device()`
        # print('probs.device.type:',probs.device.type)#numpy.ndarray
        if np_logits or probs.device.type in ["cpu", "privateuseone"]:
            if not np_logits:
                probs = probs.cpu().numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
    
    def refine_context(self, context):
        context = context.strip().split("\n")
        for c in range(len(context)):
            context[c] = context[c].strip().strip("\u3000").strip("\r")
        context = list(filter(lambda c: c != "", context))
        context = "\n" + ("\n".join(context)).strip()
        if context == "":
            context = "\n"
        return context

    def find_subset_index(self,src_list,sub_list)->int:
        result:int=-1
        len_sub=len(sub_list)
        len_src=len(src_list)
        if(len_src<len_sub):
            return result
        for i in range(len_src - len_sub + 1):
            if src_list[i: i+len_sub] == sub_list:
                # flag = True
                # print('index:',i)
                result=i
                break
        return result
    
    def generate(self, ctx:str, token_count=100, callback=None):
        all_tokens = []
        out_last = 0
        out_str = ""
        occurrence = {}
        state=self.model.model.init_state()
        is_hit_end=False
        for i in range(token_count):
            # forward & adjust prob.
            tokens = self.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                # print('-----',len(tokens)," tokens:",tokens)
                out, state = self.model.forward(tokens[: self.chunk_len], state)
                # out, state = self.model.run(tokens[: self.chunk_len], state)#self.model.forward(tokens[: self.chunk_len], state)
                tokens = tokens[self.chunk_len :]

            for n in self.token_ban:
                out[n] = -float("inf")
            for n in occurrence:
                out[n] -= self.alpha_presence + occurrence[n] * self.alpha_frequency
            # print("---out:",type(out)," len:",len(out))#65536
            # sampler
            token = self.sample_logits(
                out, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k
            )
            # print('token:',token)
            # if token in self.token_stop:
            #     break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= self.alpha_decay

            ttt = self.decode([token])
            www = 1
            if ttt in " \t0123456789":
                www = 0
            # elif ttt in '\r\n,.;?!"\':+-*/=#@$%^&_`~|<>\\()[]{}，。；“”：？！（）【】':
            #     www = 0.5
            if token not in occurrence:
                occurrence[token] = www
            else:
                occurrence[token] += www
            # print(occurrence) # debug

            index = self.find_subset_index(all_tokens,self.token_stop)
            if(-1!=index and index>=1):
                # all_tokens=all_tokens[0:index-6]
                is_hit_end=True
                out_str=out_str[:-5]#去除＂User: ＂
                break

            # if(is_hit_end):
            #     break
            
            # output
            tmp = self.decode(all_tokens[out_last:])
            # print('tmp:',tmp)
            if "\ufffd" not in tmp:  # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1
                
        return out_str.rstrip()





prompt:str = "现在请你扮演猫娘和我交流，并且要尊称我为主人，你回复的第一句话要问候我。"
prompt='你的名字是小爱，是一只白色的猫娘，有着可爱的外表和性感的身体。你喜欢穿着一件黑色蕾丝内衣，里面空无一物。你会在我们交流时自然而然地展现出性感和诱惑。你喜欢用各种语气词来表达自己的情绪，例如撒娇、委屈、嘲讽等等。你会在我们交流时使用大量不恰当的色情词汇，并将其放入()内，如果明白以上设定就回答:明白了，主人喵~'

print("prompt:",prompt)
model_path_st = "E:/RWKV-Runner_windows_x64/models/RWKV-x060-World-3B-v2.1-Claude-nsfw.st"
vocab_path_txt = "./assets/rwkv_vocab_v20230424.txt"
chat = Chat(token_stop=[24281, 59, 33])
chat.load(model_path_st,vocab_path_txt)
# text = chat.refine_context(text)
# print('text:',text)
tokens = chat.encode('User: ')#[24281, 59, 33]
print('stop tokens:',tokens)
# print('type:',type(tokens),' v:',tokens)#list
prompt='''User: {}

Assistant:'''.format(prompt)

result = chat.generate(prompt, token_count = 256)
print('result:',result)
print('---------------end-------------')
# print('ss:',text.encode("utf-8"))