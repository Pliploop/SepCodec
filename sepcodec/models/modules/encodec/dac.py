import dac
from encodec import EncodecModel
from encodec.utils import convert_audio
import torch
from torch import nn

from pytorch_lightning import LightningModule
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor





class DACEncoder(nn.Module):
    
    def __init__(self, sample_rate = 44100, frozen = True, model_bandwidth = 3, n_codebooks = 9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)
        self.sample_rate = sample_rate
        self.hop_length = self.model.hop_length
        self.frozen = frozen
        self.n_codebooks = n_codebooks
        
        if self.frozen:
            for param in self.model.parameters(): param.requires_grad = False
            self.model.eval()
        
        print(f"Model sample rate: {self.sample_rate}")
        print(f"Model hop length: {self.hop_length}")
        # dummy test
        print( "Dummy test: ", self.dummy_test()[0].shape, self.dummy_test()[1].shape,  self.dummy_test()[2].shape)

        
    def forward(self, wav):
        # wav is a batched waveform.unsqueeze if not:
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        # encoded_frames = self.model.encode(wav)
        # if self.sample_rate != 32000:
        #     self.model.encoder(wav)
        #     codes = torch.cat([encoded[0]
        #                     for encoded in encoded_frames], dim=-1)
        
        wav = self.model.preprocess(wav, self.sample_rate)
        z, codes, latents, _, _ = self.model.encode(wav)
        codes = codes.int()[:, :self.n_codebooks, :]

        return {
            'codes':codes,
            'embeddings':latents.permute(0, 2, 1),
            'quantized_embeddings': z.permute(0,2,1)
        }
        
    def dummy_test(self):
        
        sample_rate = self.sample_rate
        test_input = torch.rand(2, 1, 2*sample_rate)
        test_output = self.forward(test_input)
        
        codes = test_output['codes']
        embeddings = test_output['embeddings']
        quantized_embeddings = test_output['quantized_embeddings']
        
        return codes, embeddings, quantized_embeddings
    
    def decode(self, codes):
        z =  self.model.quantizer.from_codes(codes)[0]
        return self.model.decode(z)