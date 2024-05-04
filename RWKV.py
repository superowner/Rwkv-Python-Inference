from typing import Any, List, Union

try:
    import web_rwkv_py as wrp #web-rwkv-py==0.2.1
except ModuleNotFoundError:
    try:
        from . import web_rwkv_py as wrp
    except ImportError:
        raise ModuleNotFoundError(
            "web_rwkv_py not found, install it from https://github.com/cryscan/web-rwkv-py"
        )


class RWKV:
    def __init__(self, model_path: str, quant = 0, quant_nf4 = 0):
        self.w = {}  # fake weight
        # self.w["emb.weight"] = [0] * self.info.num_vocab
        # self.version = str(self.info.version).lower()
        self.wrp = wrp#getattr(wrp, self.version)
        # self.version = float(self.version.replace("v", ""))

        # layer = (
        #     int(s.lstrip("layer"))
        #     for s in strategy.split()
        #     for s in s.split(",")
        #     if s.startswith("layer")
        # )

        # chunk_size = (
        #     int(s.lstrip("chunk"))
        #     for s in strategy.split()
        #     for s in s.split(",")
        #     if s.startswith("chunk")
        # )

        args = {
            "path": model_path,
            "quant": quant,
            "quant_nf4": quant_nf4,
            
            # "turbo": True,
            # "token_chunk_size": 32,
            # "lora": None,
        }
        self.model = self.wrp.Model(**args)
        info = self.model.info()
        # print('--info:',info)

        #OK
        # print('model_version:',info.version)#ModelVersion
        # print('num_layer:',info.num_layer)
        # print('num_layer:',info.num_layer)
        # print('num_hidden:',info.num_hidden)
        # print('num_vocab:',info.num_vocab)
        # print('num_head:',info.num_head)
        # print('time_mix_adapter_size:',info.time_mix_adapter_size)
        # print('time_decay_adapter_size:',info.time_decay_adapter_size)

        self.w["emb.weight"] = [0] * info.num_vocab
        self.version = str(info.version).lower()
        self.version = float(self.version.replace("v", ""))

    def forward(self, tokens: List[int], state: Union[Any, None] = None,token_chunk_size =128):
        #print('type name:',type(state).__name__)#type name: State
        # _gpu_state = self.model.init_state()
        # _gpu_state.load(state)
        # print('gpu_state:',_gpu_state)
        if type(state).__name__ == "BackedState":  # memory state
            gpu_state = self.wrp.ModelState(self.model, 1)
            print('gpu_state:',gpu_state)
            gpu_state.load(state)
        else:
            gpu_state = state
        return self.model.run(tokens, gpu_state, token_chunk_size)