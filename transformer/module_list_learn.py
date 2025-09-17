import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

class ModelWithList(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [Block(), Block()]  # Plain list, NOT registered

class ModelWithModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])  # Registered properly

model_list = ModelWithList()
model_modlist = ModelWithModuleList()

print("Params in model_list:", sum(p.numel() for p in model_list.parameters()))
print("Params in model_modlist:", sum(p.numel() for p in model_modlist.parameters()))
