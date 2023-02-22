import torch 
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x.to(torch.float16)).to(torch.float32)
    
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def get_flanT5_peft(
    model_ckpt="google/flan-t5-xxl", 
    load_in_8bit=True,
    lora_r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05
    ):
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, load_in_8bit=load_in_8bit, device_map='auto') # load 8-bit flan-t5 model
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:# cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q", "v"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM)
    

    model = get_peft_model(model, config)
    return model,tokenizer
