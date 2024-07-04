# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2024/7/4
# Brief  Download model and save to pt file.

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")
model.eval()
example = torch.rand(1, 512).long()
traced_script_module = torch.jit.trace(model, example, strict=False)
traced_script_module.save("data/bert-base-chinese.pt")