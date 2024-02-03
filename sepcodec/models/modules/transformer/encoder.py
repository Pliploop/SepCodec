
import random
import torch
from torch import nn
from sepcodec.models.modules.utils.position import PositionalEncoding
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer
from pytorch_lightning import LightningModule



# class Embed(nn.Module):
#     def __init__(
#         self, embedding_behaviour, embedding_sizes, n_codebooks, card, special_tokens = ('MASK','PAD') , *args, **kwargs
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.embedding_behaviour = embedding_behaviour

#         self.embedding_sizes = embedding_sizes

#         self.n_codebooks = n_codebooks
#         self.card = card

#         self.emb = nn.ModuleList(
#             [
#                 nn.Embedding(self.card + 3, self.embedding_sizes[codebook])
#                 for codebook in range(self.n_codebooks)
#             ]
#         )

#         # +3 for pad, pattern tokens, and mask tokens

#     def forward(self, indices):
#         B, K, T = indices.shape

#         embeddings = [self.emb[k](indices[:, k, :])
#                       for k in range(K)]  # shape B,T,E
#         if self.embedding_behaviour == "sum":
#             input_ = sum(embeddings)
#         else:
#             input_ = torch.cat(embeddings, dim=-1)

#         return input_



class Encoder(nn.Module):
    

    def __init__(
        self,
        # n_codebooks=4,
        # embedding_size=[512, 256, 128, 64],
        # card=1024,
        # embedding_behaviour="concat",
        sequence_len=1024,
        layers=2,
        n_heads=8,
        d_model = 512,
        # batched_mask=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # self.n_codebooks = n_codebooks
        # self.embedding_behaviour = embedding_behaviour
        # self.embedding_size = embedding_size

        # self.card = card
        self.sequence_len = sequence_len
        self.d_model = d_model
        # self.mask_special_token = self.card + 2
        # self.pad_special_token = self.card + 3
        # self.position_encoder = position_encoder

        # if self.embedding_behaviour == "concat":
        #     self.d_model = sum(self.embedding_size)
        # else:
        #     self.d_model = self.embedding_size[0]

        self.position_encoder = PositionalEncoding(
            self.d_model, max_len=self.sequence_len
            )

        # self.emb = Embed(
        #     embedding_behaviour=self.embedding_behaviour,
        #     embedding_sizes=self.embedding_size,
        #     card=self.card,
        #     n_codebooks=self.n_codebooks,
        # )
        
        # self.linears = nn.ModuleList(
        #     [
        #         nn.Linear(self.embedding_size[codebook], self.card)
        #         for codebook in range(self.n_codebooks)
        #     ]
        # )

        self.n_heads = n_heads
        self.layers = layers

        self.transformer = None

        self.norm_in = LayerNorm(self.d_model)
        # self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        # self.mask_token = nn.Parameter(torch.randn(self.d_model))
        
        # create a dense embedding layer for class conditioning. To adapt later on
        # self.class_conditioning = nn.Embedding(5, self.d_model)
        
        # self.encoder_mask_emb = nn.Parameter(torch.FloatTensor(self.d_model).uniform_())
        # self.batched_mask = batched_mask

        self.norm = LayerNorm(self.d_model)
        self.transformer_layer = TransformerEncoderLayer(
            self.d_model, self.n_heads, activation="gelu", batch_first=True, dropout=0.1
        )
        self.transformer = TransformerEncoder(
            self.transformer_layer, self.layers, norm=self.norm
        )

        self.first_run = True
        
        # self.proj = nn.Linear(128, self.d_model)
        self.decoder_proj = nn.Linear(self.d_model, self.d_model)

    def adapt_sequence_len(self, new_sequence_len):
        self.sequence_len = new_sequence_len

    # def embed(self, codes, embeddings=None, use_embeddings=False):
    #     if not use_embeddings:
    #         original_embeddings = self.emb(codes)
    #     else:
    #         original_embeddings = self.proj(embeddings)
    #     return original_embeddings
            

    def forward(self, codes, original_embeddings = None, conditioning = None, padding_mask=None, mask_before=False, mask=True, use_embeddings = True, embeddings = None):
        # indices is of shape B,n_q,T
        B, K, T = codes.shape

        if original_embeddings is None:
            original_embeddings = self.embed(codes, embeddings=embeddings, use_embeddings=use_embeddings)            
            
            if self.first_run:
                print("encodec embeddings: {}".format(original_embeddings.shape))
            
            
            
        
        # if padding_mask is None:
        #     padding_mask = torch.zeros(B, T, device=codes.device)
        # padding_mask = torch.cat(
        #     [torch.zeros(B, 1, device=padding_mask.device), padding_mask], dim=1)
        
        # print(class_token.shape)
        # input_ = torch.cat([original_embeddings, class_token], dim=1)
        # concat here but could also sum or classifier free guidance
        input_ = self.position_encoder(original_embeddings)  # B,T+1,d_model
            

        # if not mask_before and mask:
        #     input_, masked_idx, retained_idx, retained_padding_mask, codes_mask, contrastive_matrix, contrastive_matrix_blackout, contrastive_matrix_masked = self.mask_after(
        #         x=input_, padding_mask=padding_mask, codes=codes, contrastive_matrix=contrastive_matrix
        #     )
        # B, T//mask_proportion +1, d_model

        # if not mask:
        #     codes_mask = torch.zeros_like(codes, device=codes.device)
        #     retained_padding_mask = padding_mask
        #     retained_idx = [list(range(T)) for k in range(K)]
        #     masked_idx = []
        
        # codes = codes.clone()
        # codes[codes_mask] = self.mask_special_token

        # if self.first_run:
        #     print("============== codes_masking ==============")
        #     print(codes_mask)
        #     print(codes_mask.shape)
        #     print(f"{codes_mask.sum()} tokens were masked with random masking")
        #     print(codes)

        # shape B,T+1,d_model
        
        input_ = self.norm_in(input_)
        # encoded = self.transformer(
        #     input_, src_key_padding_mask=retained_padding_mask)
        
        encoded = self.transformer( input_)
        # shape B,T+1,d_model

        if self.first_run:
            print("shape coming out of encoder: ============")
            print(encoded.shape)
            
        output_ = self.decoder_proj(encoded) ## to not share the output latent space with the decoder input space
        
        if self.first_run:
            print("========= All outputs for the encoder========")
            print("-------- masked output with class token --------")
            print(output_.shape)
            print(
                "-------- unmasked output with class token and original embeddingd without class token --------"
            )
            print(original_embeddings.shape)
            # print("-------- codes_mask.shape ---------")
            # print(codes_mask.shape)
            # print("------ padding_mask.shape ----------")
            # print(padding_mask.shape)

        self.first_run = False

        

        # return output_, codes_mask, padding_mask
        return {"codes" : codes,
                "embeddings" : output_}


class VanillaEncoder(Encoder):
    def __init__(
        self,
        sequence_len=2048,
        layers=2,
        n_heads=8,
        d_model = 512,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            sequence_len,
            layers,
            n_heads,
            d_model,
            *args,
            **kwargs,
        )
        self.norm = LayerNorm(self.d_model)
        self.transformer_layer = TransformerEncoderLayer(
            self.d_model, self.n_heads, activation="gelu", batch_first=True
        )
        self.transformer = TransformerEncoder(
            self.transformer_layer, self.layers, norm=self.norm
        )
