
import random
import torch
from torch import nn
from sepcodec.models.modules.utils.position import PositionalEncoding
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer
from pytorch_lightning import LightningModule



class Embed(nn.Module):
    def __init__(
        self, embedding_behaviour, embedding_sizes, n_codebooks, card, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_behaviour = embedding_behaviour

        self.embedding_sizes = embedding_sizes

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

        return input_



class Encoder(nn.Module):
    """ "Default transformer encoder. Default behaviour is according to encodecMAE (or similar):

        sum embeddings of all tokens and conduct masking afterwards (with or without codebook pattern generator).

    Other possible behaviours :

        Pattern + Masking before summing embeddings. Meaning the masking mask would include all embeddings. Allows for structured masking patterns like in patchout
        Pattern + Masking before flattening embeddings. Allows for structured patterns in masking and discarding embeddings *BUT* results in 4x longer sequence


    """

    def __init__(
        self,
        n_codebooks=4,
        embedding_size=[512, 256, 128, 64],
        card=1024,
        embedding_behaviour="concat",
        sequence_len=1024,
        layers=2,
        n_heads=8,
        batched_mask=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.n_codebooks = n_codebooks
        self.embedding_behaviour = embedding_behaviour
        self.embedding_size = embedding_size

        self.card = card
        self.sequence_len = sequence_len
        self.mask_special_token = self.card + 2
        self.pad_special_token = self.card + 3
        # self.position_encoder = position_encoder

        if self.embedding_behaviour == "concat":
            self.d_model = sum(self.embedding_size)
        else:
            self.d_model = self.embedding_size[0]

        self.position_encoder = PositionalEncoding(
            self.d_model, max_len=self.sequence_len
            )

        self.emb = Embed(
            embedding_behaviour=self.embedding_behaviour,
            embedding_sizes=self.embedding_size,
            card=self.card,
            n_codebooks=self.n_codebooks,
        )
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.embedding_size[codebook], self.card)
                for codebook in range(self.n_codebooks)
            ]
        )

        self.n_heads = n_heads
        self.layers = layers

        self.transformer = None

        self.norm_in = LayerNorm(self.d_model)
        # self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        # self.mask_token = nn.Parameter(torch.randn(self.d_model))
        
        # create a dense embedding layer for class conditioning. To adapt later on
        self.class_conditioning = nn.Embedding(5, self.d_model)
        
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
        
        self.proj = nn.Linear(128, self.d_model)
        self.decoder_proj = nn.Linear(self.d_model, self.d_model)

    def adapt_sequence_len(self, new_sequence_len):
        self.sequence_len = new_sequence_len

    def forward(self, codes, conditioning = None, padding_mask=None, mask_before=False, mask=True, embeddings=None, use_embeddings=False):
        # indices is of shape B,n_q,T
        B, K, T = codes.shape

        if not use_embeddings:
            original_embeddings = self.emb(codes)  # B,T,d_model
        else:
            original_embeddings = self.proj(embeddings) # B,T,d_model
            
            if self.first_run:
                print("encodec embeddings: {}".format(original_embeddings.shape))
            
            
            
        
        if conditioning is not None:
            class_token = self.class_conditioning(conditioning)
        else:
            class_token = torch.zeros(B, 1, self.d_model, device=codes.device)
        
        # if padding_mask is None:
        #     padding_mask = torch.zeros(B, T, device=codes.device)
        # padding_mask = torch.cat(
        #     [torch.zeros(B, 1, device=padding_mask.device), padding_mask], dim=1)
        input_ = torch.cat([original_embeddings, class_token], dim=1)
        # concat here but could also sum or classifier free guidance
        input_ = self.position_encoder(input_)  # B,T+1,d_model
            

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
        return output_


    # def unmask(
    #     self,
    #     embeddings,
    #     original_embeddings,
    #     masked_idx,
    #     retained_idx,
    #     retained_padding_mask,
    # ):
    #     class_token = embeddings[:, 0, :].unsqueeze(1)
    #     without_class_token = embeddings[:, 1:, :]
        
    #     B, T, _ = original_embeddings.shape

    #     all_masked = self.mask_token.expand(B, T, -1).clone()

    #     if self.first_run:
    #         print("=========== Masked without embeddings shape ========")
    #         print(all_masked.shape)

    #     for i, (cur_feat, ridx, midx) in enumerate(
    #         zip(without_class_token, retained_idx,
    #             masked_idx)
    #     ):
    #         all_masked[i, ridx] = cur_feat.clone()
    #         all_masked[i, midx] = self.mask_token.clone().expand(len(midx), -1)

    #     if self.first_run:
    #         print("========= all_masked.shape ==========")
    #         print(all_masked.shape)
    #         print("class token =============")
    #         print(class_token.shape)

    #     all_masked = torch.cat([class_token, all_masked], dim=1)

    #     return all_masked

    # def mask_after(self, x, padding_mask, codes, contrastive_matrix, mask_gap = 15):
        """creates a mask for the input. note that the same amount of tokens must be masked in each batch (because of matrices). so either:
        - mask a precise amount of tokens
        - create a batch mask (easier)
        """
        # sample binomial probability

        # note that here teh contrastive matrix already has the class token included
        
        class_token = x[:, 0, :]
        x = x[:, 1:, :]
        
        class_padding_mask = padding_mask[:, 0]
        padding_mask = padding_mask[:, 1:]
        

        B, T, _ = x.shape
        num_retained_tokens = int((1 - self.mask_p) * T)
        num_retained_tokens = max(1, num_retained_tokens)
        if contrastive_matrix:
            matrix = contrastive_matrix[0]
            masked_matrix_blackout = matrix.clone()
        else:
            matrix = None
            masked_matrix_blackout = None

        if contrastive_matrix:
            _, _, T_mat, _ = contrastive_matrix[1]

        if self.first_run:
            print(f"masking after with masking proba : {self.mask_p}")
            print(x.shape)
            print(self.encoder_mask_emb.shape)

        # used to compute loss over masked tokens. because this is after, mask by columns
        codes_mask = torch.zeros_like(codes, device=codes.device)

        retained_idx = []
        masked_idx = []

        for i in range(B):
            idx = list(range(T))
            random.shuffle(idx)
            cur_retained_idx = idx[:num_retained_tokens]
            retained_idx.append(cur_retained_idx)
            cur_masked_idx = idx[num_retained_tokens:]
            masked_idx.append(cur_masked_idx)
            x[i, cur_masked_idx, :] = self.encoder_mask_emb.clone().expand(
                len(cur_masked_idx), -1)
            codes_mask[i, :, cur_masked_idx] = 1

            cur_masked_idx = masked_idx[i]
            

            if contrastive_matrix and T_mat == T +1:
                cur_masked_idx_mat = [k+1 + i*(T_mat+1)
                                      for k in cur_masked_idx]
                masked_matrix_blackout[cur_masked_idx_mat, :] = -1
                masked_matrix_blackout[:, cur_masked_idx_mat] = -1

        if self.batched_mask:
            x = x[:, retained_idx[0]]
            retained_padding_mask = padding_mask[:, retained_idx[0]]
        else:
            new_x = []
            retained_padding_mask = []
            for i in range(B):
                new_x.append(x[i, retained_idx[i]])
                retained_padding_mask.append(padding_mask[i, retained_idx[i]])
            x = torch.stack(new_x, dim=0)
            retained_padding_mask = torch.stack(retained_padding_mask, dim=0)

        codes_mask = (codes_mask == 1)
        if contrastive_matrix:
            index = (masked_matrix_blackout != -1)[0].nonzero().squeeze()
            masked_matrix = matrix[index, :][:, index]
        else:
            masked_matrix = None
            
        x = torch.cat([class_token.unsqueeze(1), x], dim=1)
        retained_padding_mask = torch.cat([class_padding_mask.unsqueeze(1), retained_padding_mask], dim=1)

        return x, masked_idx, retained_idx, retained_padding_mask, codes_mask, contrastive_matrix, masked_matrix_blackout, masked_matrix


class VanillaEncoder(Encoder):
    def __init__(
        self,
        n_codebooks=4,
        embedding_size=[512, 256, 128, 64],
        card=1024,
        embedding_behaviour="concat",
        position_encoder="sinusoidal",
        sequence_len=2048,
        layers=6,
        n_heads=8,
        p=0.5,
        batched_mask=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            n_codebooks,
            embedding_size,
            card,
            embedding_behaviour,
            position_encoder,
            sequence_len,
            layers,
            n_heads,
            p,
            batched_mask,
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
