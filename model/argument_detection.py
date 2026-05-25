import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM
from .basemodelshared import LlamaSingleton
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

class argumentDetection(nn.Module):
    def __init__(self, config):
        super(argumentDetection, self).__init__()
        self.config = config
        # Load the LLaMA model
        llama_instance = LlamaSingleton(config)  # Get the singleton instance
        self.llama = copy.deepcopy(llama_instance.llama) 
        self.embedding_dim = self.llama.config.hidden_size
        
        # Apply LoRA to the model
        lora_config = LoraConfig(
            r=8,  # Rank for LoRA
            lora_alpha=16,  # Scaling factor for LoRA
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear"  # Apply to attention projection layers
        )
        
        # Apply the LoRA configuration to the LLaMA model
        self.llama = get_peft_model(self.llama, lora_config)
        
        # The classifier layer now uses self.embedding_dim * 2 because embeddings are concatenated
        self.classifier = nn.Linear(self.embedding_dim * 2, config.args_num, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, segment_ids, input_mask, offset, metadata, unseen_metadata, trigger, ner, gold_args):
        # Forward pass through LLaMA
        sequence_output = self.llama(input_ids, attention_mask=input_mask).last_hidden_state
        new_logits = None
        new_label = []
        
        # Process each sentence
        for i in range(len(ner)):
            for start, end in ner[i]:
                # Get the embeddings for the start and end tokens
                embedding = sequence_output[i][[start + 1, end]].view(-1, self.embedding_dim * 2)
                embedding = self.dropout(embedding)
                logits = self.classifier(embedding)
                
                # Zero out logits for unseen arguments
                one_trigger = trigger[i]
                unseen_args = unseen_metadata[one_trigger]
                logits[:, unseen_args] = 0
                
                # Get the corresponding label
                label = labels[i][start + 1]
                new_label.append(label)
                
                # Concatenate logits for all tokens
                if new_logits is None:
                    new_logits = logits
                else:
                    new_logits = torch.cat([new_logits, logits], dim=0)

        new_label = torch.tensor(new_label).cuda()
        
        # Calculate the loss
        loss = self.criterion(new_logits, new_label)
        return loss

    def get_res(self, input_ids, segment_ids, input_mask, ner):
        # Forward pass through LLaMA
        sequence_output = self.llama(input_ids, attention_mask=input_mask).last_hidden_state
        res_logits = []

        # Process each sentence
        for i in range(len(ner)):
            one_logits = None
            for start, end in ner[i]:
                # Get the embeddings for the start and end tokens
                embedding = sequence_output[i][[start + 1, end]].view(-1, self.embedding_dim * 2)
                embedding = self.dropout(embedding)
                logits = self.classifier(embedding)

                # Concatenate logits for all tokens in the current sentence
                if one_logits is None:
                    one_logits = logits
                else:
                    one_logits = torch.cat([one_logits, logits], dim=0)
            
            res_logits.append(one_logits)
        return res_logits

    def get_feature(self, input_ids, segment_ids, input_mask):
        # Forward pass through LLaMA
        sequence_output = self.llama(input_ids, attention_mask=input_mask).last_hidden_state
        feature = self.dropout(sequence_output)
        feature = feature.view((1, -1))
        return feature