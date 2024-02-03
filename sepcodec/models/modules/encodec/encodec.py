from encodec import EncodecModel
from encodec.utils import convert_audio
import torch
from torch import nn

from pytorch_lightning import LightningModule
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor



class Encodec(nn.Module):

    def __init__(self,  sample_rate=32000, frozen=True, model_bandwidth=3, n_codebooks = 6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frozen = frozen
        self.sample_rate = sample_rate
        
        self.model_bandwidth = model_bandwidth
        if sample_rate == 24000:
            self.model = EncodecModel.encodec_model_24khz()
            self.hop_length = self.model.encoder.hop_length
            self.model.set_target_bandwidth(self.model_bandwidth)
            self.card = 1024
            self.fps = 75
        elif sample_rate == 48000:
            self.model = EncodecModel.encodec_model_48khz()
            self.hop_length = self.model.encoder.hop_length
            self.model.set_target_bandwidth(self.model_bandwidth)
            self.fps = 75
        elif sample_rate == 32000:
            self.model = AutoModel.from_pretrained("facebook/encodec_32khz")
            self.hop_length = sample_rate // 50 # model outputs 50 frames per second, which is 640 hop length at 32khz
            self.card = 2048
            self.fps = 50
            
        self.n_codebooks = n_codebooks
        print(f"Model sample rate: {self.sample_rate}")
        print(f"Model hop length: {self.hop_length}")
        # dummy test
        print( "Dummy test: ", self.dummy_test()[0].shape, self.dummy_test()[1].shape, self.dummy_test()[2].shape)
        
        
        
        
        if self.frozen:
            for param in self.model.parameters(): param.requires_grad = False
            self.model.eval()
        

    def forward(self, wav):
        # wav is a batched waveform.unsqueeze if not:
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        # encoded_frames = self.model.encode(wav)
        # if self.sample_rate != 32000:
        #     self.model.encoder(wav)
        #     codes = torch.cat([encoded[0]
        #                     for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        # else:
            # codes = encoded_frames.audio_codes.squeeze(0)
        embeddings = self.model.encoder(wav)
        codes = self.model.quantizer.encode(embeddings).int()
        quantized_embeddings = self.model.quantizer.decode(codes)
        embeddings = embeddings.permute(0, 2, 1)
        quantized_embeddings = quantized_embeddings.permute(0, 2, 1)
            # codes = torch.zeros(1, 1, 1)
        codes = codes[:, :self.n_codebooks, :]
        
        
        return {
            'codes':codes,
            'embeddings':embeddings,
            'quantized_embeddings': quantized_embeddings
        }

    def get_encodec_output(self,wav):
        
        output = self(wav)
        codes = output['codes']
        embeddings = output['embeddings']
        quantized_embeddings = output['quantized_embeddings']
        
        return codes,embeddings, quantized_embeddings
    
    
    
    
    def dummy_test(self):
        
        dummy_data = torch.randn(2,1,self.sample_rate*2)
        return self.get_encodec_output(dummy_data)