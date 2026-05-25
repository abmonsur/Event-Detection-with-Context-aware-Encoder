import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM
from .basemodelshared import LlamaSingleton
from peft import get_peft_model, LoraConfig, TaskType
import warnings
from torchcrf import CRF

warnings.filterwarnings('ignore')

class entityDetection(nn.Module):
    def __init__(self, config, rnn_dim=128):
        super(entityDetection, self).__init__()
        
        # Load the LLaMA model
        # llama_instance = LlamaSingleton(config)  # Get the singleton instance
        self.llama = LlamaSingleton(config).llama
        
        # # Apply LoRA to the model
        # lora_config = LoraConfig(
        #     r=8,  # Rank for LoRA
        #     lora_alpha=16,  # Scaling factor for LoRA
        #     lora_dropout=0.2,
        #     bias="lora_only",
        #     target_modules="all-linear"  # Apply to attention projection layers
        # )
        
        # # Apply the LoRA configuration to the LLaMA model
        # self.llama = get_peft_model(self.llama, lora_config)
        
        # Additional layers for the entity detection task
        self.dropout = nn.Dropout(0.2)
        self.birnn = nn.LSTM(768, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(rnn_dim * 2, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, labels, token_type_ids=None, input_mask=None):
        # Forward pass through LLaMA
        outputs = self.llama(input_ids, attention_mask=input_mask)
        sequence_output = outputs.last_hidden_state
        
        # Pass through Bi-RNN
        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        # Classify the output
        emissions = self.classifier(sequence_output)
        
        # Compute CRF loss
        loss = -1 * self.crf(emissions, labels, mask=input_mask.byte())
        return loss

    def get_res(self, input_ids, token_type_ids=None, input_mask=None):
        # Forward pass through LLaMA
        outputs = self.llama(input_ids, attention_mask=input_mask)
        sequence_output = outputs.last_hidden_state
        
        # Pass through Bi-RNN
        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        # Classify the output
        emissions = self.classifier(sequence_output)
        
        # Decode using CRF layer
        res = self.crf.decode(emissions, input_mask.byte())
        return res