
from typing import Any, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
import wandb
from pytorch_lightning.cli import OptimizerCallable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


class SepCodec(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        # decoder: nn.Module,
        encodec: nn.Module,
        optimizer: OptimizerCallable = None,
        n_codebooks=4,
        sequence_len=1024,
        debug=False,
        adapt_sequence_len=True,
        reduce_lr_monitor='masked loss',
        use_embeddings=True,
        resume_from_checkpoint=None,
        checkpoint_path=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = encodec
        self.transformer_encoder = encoder
        # self.transformer_decoder = decoder
        self.d_model = self.transformer_encoder.d_model
        self.sequence_len = sequence_len
        self.adapt_sequence_len = adapt_sequence_len
        self.n_codebooks = n_codebooks
        self.debug = debug

        self.reduce_lr_monitor = reduce_lr_monitor

        self.use_encodec_embeddings = use_embeddings

        self.optimizer = optimizer

        self.card = self.transformer_encoder.card
        self.mask_special_token = self.card + 2
        self.pad_special_token = self.card+3

        if resume_from_checkpoint:
            self.load_from_checkpoint(resume_from_checkpoint)

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)['state_dict']
            self.load_state_dict(state_dict, strict=False)

        self.dummy_test()

        self.conditioning_mech = "class_idx"

    def dummy_test(self):
        sample_rate = self.encodec.sample_rate

        test_input = {
            'mix': torch.rand(1, 2, 2*sample_rate),
            'target_stem': torch.rand(1, 2, 2*sample_rate),
            'conditioning': {
                'class_idx': torch.randint(0, 5, (1,)),
                'audio_query': torch.rand(1, 2, 2*sample_rate),
                'cluster_idx': torch.randint(0, 10, (1,))
            }
        }

        # print some stuff for sanity check
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input sample rate: {sample_rate}")
        test_output = self.forward(test_input)

        # print all shapes in output dict
        print("Test output shapes:")
        for key in test_output.keys():
            print(f"{key}: {test_output[key].shape}")

        return test_input, test_output

    def forward(self, x):

        mix = x['mix']
        target_stem = x['target_stem']
        conditioning = x['conditioning'][self.conditioning_mech]

        encodec_mix = self.encodec(mix)
        encodec_target = self.encodec(target_stem)

        mix_embeddings = encodec_mix['embeddings']
        mix_codes = encodec_mix['codes']

        target_embeddings = encodec_target['embeddings']
        target_codes = encodec_target['codes']


        encoded_mix = self.transformer_encoder(mix_embeddings, conditioning=conditioning, mask=False,
                                               embeddings=mix_embeddings, use_embeddings=self.use_encodec_embeddings, padding_mask=padding_mask)

        return {
            'encoded_mix': encoded_mix,
            'mix_codes': mix_codes,
            'target_codes': target_codes,
            'target_embeddings': target_embeddings
        }


    def greedy_sampling(self,logits, temperature=1.0):
        return torch.argmax(logits, dim=-1)
    
    def nucleus_sampling(self,logits, p=0.9):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        return torch.multinomial(F.softmax(logits, dim=-1), 1)
    
    def top_k_sampling(self,logits, k=50):
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.multinomial(F.softmax(logits, dim=-1) * (logits >= min_values).float(), 1)
    
    def temperature_sampling(self,logits, temperature=1.0):
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
    
    def sample(self,logits, sampling_strategy='greedy', temperature=1.0, p=0.9, k=50):
        if sampling_strategy == 'greedy':
            return self.greedy_sampling(logits, temperature)
        elif sampling_strategy == 'nucleus':
            return self.nucleus_sampling(logits, p)
        elif sampling_strategy == 'top_k':
            return self.top_k_sampling(logits, k)
        elif sampling_strategy == 'temperature':
            return self.temperature_sampling(logits, temperature)
        else:
            raise ValueError(f"Sampling strategy {sampling_strategy} not supported")