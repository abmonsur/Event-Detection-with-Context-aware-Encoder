import torch
import torch.nn as nn
import copy
import os
from transformers import AutoModel, AutoConfig
from .basemodelshared import LlamaSingleton
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_initializer='xavier', **kwargs):
        super(LayerNorm, self).__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal_(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            cond = self.hidden_dense(cond) if self.hidden_units is not None else cond
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)
            beta = self.beta_dense(cond) + self.beta if self.center else None
            gamma = self.gamma_dense(cond) + self.gamma if self.scale else None
        else:
            beta, gamma = (self.beta if self.center else None), (self.gamma if self.scale else None)

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1, keepdim=True)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
            std = torch.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma if self.scale else outputs
        if self.center:
            outputs = outputs + beta if self.center else outputs
        return outputs

class triggerEncoder(nn.Module):
    def __init__(self, config):
        super(triggerEncoder, self).__init__()
        self.config = config
        self.last_k_attention = config.last_k_attention

        # Load LLaMA model and apply LoRA
        self.llama = AutoModel.from_pretrained(config.llama_path)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.2,
            target_modules="all-linear"
        )
        self.llama = get_peft_model(self.llama, lora_config)

        self.embedding_dim = self.config.embedding_dim
        self.hidden_dim = self.llama.config.hidden_size
        self.drop = nn.Dropout(0.2)

        self.linear_transform = nn.Linear(self.hidden_dim, self.config.hidden_dim)

        # FiLM generator: takes sentence embedding and outputs gamma and beta
        self.film_generator = nn.Linear(self.config.hidden_dim, 2 * self.config.hidden_dim)

        self.layer_normalization = nn.LayerNorm(self.config.hidden_dim)

    def get_attention(self, input_ids, attention_mask, segment_ids):
        output = self.llama(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attention = output.attentions
        now_attention = 0
        for i in range(self.last_k_attention):
            now_layer_att = attention[-i - 1]
            now_layer_att = torch.mean(now_layer_att, 1)
            res_att = now_layer_att / (torch.sum(now_layer_att, dim=-1, keepdim=True) + 1e-9)
            now_attention += res_att
        avg_layer_att = now_attention / self.last_k_attention
        return avg_layer_att

    def get_feature(self, sentence_ids, input_ids, attention_mask, segment_ids):
        feature = self.llama(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        feature = feature.hidden_states[-1]
        seq_output = self.drop(feature)
        seq_output = self.linear_transform(seq_output)
        output = F.gelu(seq_output)
        feature = self.layer_normalization(output)
        feature = feature.view((1, -1))
        return feature

    def forward(self, sentence_ids, input_ids, attention_mask, segment_ids):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        # Project to your hidden_dim
        token_reps = self.linear_transform(self.drop(hidden_states))  # (batch, seq_len, dim)
        token_reps = F.gelu(token_reps)

        # Pool sentence embedding
        sentence_emb = token_reps.mean(dim=1)  # (batch, dim)

        # Generate gamma and beta for FiLM
        gamma_beta = self.film_generator(sentence_emb)  # (batch, 2*dim)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # each is (batch, dim)

        # Expand for broadcasting
        gamma = gamma.unsqueeze(1)  # (batch, 1, dim)
        beta = beta.unsqueeze(1)    # (batch, 1, dim)

        # Apply FiLM modulation
        modulated = gamma * token_reps + beta  # (batch, seq_len, dim)

        # Final normalization
        output = self.layer_normalization(modulated)
        return output