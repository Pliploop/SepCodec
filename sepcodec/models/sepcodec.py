
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




class Embed(nn.Module):
    def __init__(
        self, embedding_behaviour = 'concat', embedding_sizes = [256,256,128,128,128,128,64,64,64], n_codebooks = 9, card = 1024 , *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_behaviour = embedding_behaviour

        self.embedding_sizes = embedding_sizes
        self.d_model = sum(embedding_sizes)
        
        
        self.n_codebooks = n_codebooks
        self.card = card
        

        self.emb = nn.ModuleList(
            [
                nn.Embedding(self.card + 3, self.embedding_sizes[codebook])
                for codebook in range(self.n_codebooks)
            ]
        )

        # +3 for pad, pattern tokens, and mask tokens

    def forward(self, indices):
        B, K, T = indices.shape

        embeddings = [self.emb[k](indices[:, k, :])
                      for k in range(K)]  # shape B,T,E
        if self.embedding_behaviour == "sum":
            input_ = sum(embeddings)
        else:
            input_ = torch.cat(embeddings, dim=-1)

        return input_.permute(0,2,1)

class SepCodec(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encodec: nn.Module,
        embed: nn.Module,
        optimizer: OptimizerCallable = None,
        n_codebooks=9,
        sequence_len=1024,
        d_model = 512,
        card = 1024,
        debug=False,
        adapt_sequence_len=True,
        reduce_lr_monitor='masked loss',
        resume_from_checkpoint=None,
        checkpoint_path=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = encodec
        self.transformer_encoder = encoder
        self.transformer_decoder = decoder
        self.embed = embed
        # self.d_model = self.transformer_encoder.d_model
        
        self.d_model = d_model
        self.sequence_len = sequence_len
        self.adapt_sequence_len = adapt_sequence_len
        self.n_codebooks = n_codebooks
        self.debug = debug

        self.reduce_lr_monitor = reduce_lr_monitor


        self.optimizer = optimizer

        self.card = card
        self.mask_special_token = self.card+1
        self.pad_special_token = self.card+2
        
        #create a mask embedding as a learnable parameter
        # self.mask_embedding = nn.Parameter(torch.randn(self.d_model))

        if resume_from_checkpoint:
            self.load_from_checkpoint(resume_from_checkpoint)

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)['state_dict']
            self.load_state_dict(state_dict, strict=False)


        # self.quantized_embedding_size = self.encodec.dummy_test()[2].shape[-1]

        self.in_proj = nn.Conv1d(self.embed.d_model, self.d_model, 1)

        
        self.conditioning_mech = "class_idx"
         # create a dense embedding layer for class conditioning. To adapt later on
        self.class_conditioning = nn.Embedding(5, self.d_model)
        self.lin_out = nn.Linear(self.d_model, self.n_codebooks * self.card)
        
        self.dummy_test()
        
    
        
        


    def dummy_test(self):
        sample_rate = self.encodec.sample_rate

        test_input = {
            'mix': torch.rand(2, 1, 2*sample_rate),
            'target_stem': torch.rand(2, 1, 2*sample_rate),
            'conditioning': {
                'class_idx': torch.randint(0, 5, (2,1)),
                'audio_query': torch.rand(2, 1, 2*sample_rate),
                'cluster_idx': torch.randint(0, 10, (2,1))
            }
        }

        # print some stuff for sanity check
        print("Test input shapes:")
        for key in test_input.keys():
            if key == 'conditioning':
                for sub_key in test_input[key].keys():
                    print(f"{key}/{sub_key}: {test_input[key][sub_key].shape}")
            else:
                print(f"{key}: {test_input[key].shape}")
            
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

        # mix_quantized_embeddings = encodec_mix['quantized_embeddings']
        # we can't use the quantized embeddings yet because we need to mask the codes and can't do that without the weight dictionary
        mix_codes = encodec_mix['codes']
        mix_quantized_embeddings = self.embed(mix_codes)
        B,K,T = mix_codes.shape

        # target_quantized_embeddings = encodec_target['quantized_embeddings']
        
        
        if conditioning is not None:
            class_token = self.class_conditioning(conditioning)
        else:
            class_token = torch.zeros(B, 1, self.d_model, device=mix_codes.device)
        
        
        
        pre_encoder_mix = self.in_proj(mix_quantized_embeddings).permute(0,2,1)
        pre_encoder_mix = torch.cat([class_token, pre_encoder_mix], dim=1)
        encoder_out = self.transformer_encoder(mix_codes, original_embeddings = pre_encoder_mix, conditioning=conditioning, mask=False)
        
        encoder_embeddings = encoder_out['embeddings']
        encoder_codes = encoder_out['codes']
        
        
        target_codes = encodec_target['codes']
        corrupted_codes = self.corrupt_codes(target_codes, p = 0.1)
        target_quantized_embeddings = self.embed(corrupted_codes)
        pre_decoder_target = self.in_proj(target_quantized_embeddings).permute(0,2,1)
        pre_decoder_target = torch.cat([class_token, pre_decoder_target], dim=1)
        
        # append the target embeddings to the encoder embeddings and the target codes to the encoder codes -> for now we're using a cross-attention mechanism but this is a viable option
        # pre_decoder_embeddings = torch.cat([encoder_embeddings, corrupted_target_embeddings], dim=1)
        
        # into decoder
        decoder_out = self.transformer_decoder(corrupted_codes, original_embeddings = pre_decoder_target, mix_embeddings = encoder_embeddings, conditioning=conditioning)
        # outputs of decoder should only be the targets. remember to remove the rest in the decoder
        decoded = decoder_out['decoded'] # shape should be (batch_size, codebooks, sequence_len, card)
        decoder_logits = self.logits_to_crossentropy_input(decoded) # [B,card,n_codebooks,T]
        #reshape logits to [B,n_codebooks,T,card] for sampling
        sampling_logits = decoder_logits.permute(0,2,3,1)
        decoded_codes = self.greedy_sampling(sampling_logits)
        
        
        print(decoded_codes.shape)
        decoded_audio = self.encodec.decode(decoded_codes)
        # decode using encodec
        
        
        return {
            'mix_codes': mix_codes, # [B,K,T]
            'target_codes': target_codes, # [B,K,T]
            'corrupted_target_codes': corrupted_codes, # [B,K,T]
            'decoded_audio': decoded_audio[:,:,:mix.shape[-1]], # [B,T]
            'decoded_codes': decoded_codes, # [B,K,T]
            'decoded_embeddings': decoded,  # [B,T,d]
            'decoded_logits': decoder_logits    # [B,card,K,T]
        }

    def corrupt_codes(self, codes, p = 0.1):
        # creates a binary mask of the same shape as codes with probability p and replaces the values with self.mask_special_token
        #note that codes is an integer tensor so don't use _like functions
        mask = torch.rand(codes.shape) < p
        corrupted_codes = codes.clone()
        corrupted_codes[mask] = self.mask_special_token
        return corrupted_codes
    
    def logits_to_crossentropy_input(self, logits):
        #logits are of shape [B,T,d]
        logits = self.lin_out(logits) # [B,T,d] -> [B,T,n_codebooks*card]
        logits = logits.view(logits.shape[0], logits.shape[1], self.n_codebooks, self.card) # [B,T,n_codebooks*card] -> [B,T,n_codebooks,card]
        logits = logits.permute(0,3,2,1) # [B,T,n_codebooks,card] -> [B,card,n_codebooks,T]
        
        return logits
        
        

    def greedy_sampling(self,logits, temperature=1.0):
        ## normally this should be the default as we're not going for diversity or creativity here
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