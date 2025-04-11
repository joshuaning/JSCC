# Following https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
# and https://www.youtube.com/watch?v=ISNdQcPhsts&ab_channel=UmarJamil (with modifications)

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # normalization
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix of (seq_len, d_model), 
        # every word (seq_len of words) is a d_model vec (embedding size)
        pe = torch.zeros(seq_len, d_model)

        #represent position of the word in the sentence, size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-1 * math.log(10000) / d_model))

        #sin to even positions, cos to odd positions

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # consider the batch of sentences
        pe = pe.unsqueeze(0)    #(1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
            


class LayerNormalization(nn.Module):

    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std =  x.std (dim = -1, keepdim=True)

        return self.weights * (x - mean) / (std + self.eps) + self.bias
    

# This is a block
class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)     # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)     # W2 and B2

    def forward(self,x):
        # input sentence (batch, seq_len, d_model)
        # linear1: --> (batch, seq_len, d_ff) --> linear2: --> (batch, seq_len, d_model)

        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# d_model have to be divisible by h (# of heads)
# this is a block
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divisible by h"
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)      #Wq
        self.w_k = nn.Linear(d_model, d_model)      #Wk
        self.w_v = nn.Linear(d_model, d_model)      #Wv
        self.w_o = nn.Linear(d_model, d_model)      #Wo

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        #(batch, h, seq_len. d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, query.size(1), -1, -1)
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1)   #(batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q)         #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)           #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)         #(batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        #(batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) -->(batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    #sublayer is the output of the next layer
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # this just makes 2 of the ResidualConnection
        # self.residual_connections = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout)])
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range (2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)          #each layer is a encoder
        return self.norm(x)
        
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward, dropout: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range (3)])

    def forward(self, x, encoder_x, src_mask, tgt_mask):
                                                #target mask is decoder mask
    
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))

        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_x, encoder_x, src_mask))

        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
        
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_x, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_x, src_mask, tgt_mask)     #each layer is a decoder
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self,x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)

        return self.proj(x)


### --- This section is dedicated for Channel related netowrks --- ###

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x

class ChannelEncoder(nn.Module):
    def __init__(self, d_model, l1_size=256, out_size=16):
        super().__init__()
        
        self.l1 = nn.Sequential(nn.Linear(d_model, l1_size),
                                nn.ReLU(inplace=True))
        self.l2 = nn.Linear(l1_size, out_size)

    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)

        return PowerNormalize(x)


class ChannelDecoder(nn.Module):
    def __init__(self, in_size, l1_size, l2_size):
        super().__init__()

        self.l1 = nn.Linear(in_size, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.l3 = nn.Linear(l2_size, l1_size)

        self.layernorm = nn.LayerNorm(l1_size, eps=1e-6)

    def forward(self, x):
        x1 = self.l1(x)
        x = F.relu(x1)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)

        output = self.layernorm(x1 + x)

        return output
    
def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

#AWGN Channel, different from the original implementation, not sure if it will work
class AWGNChannel(nn.Module):

    def __init__(self, noise_std, device):
        super().__init__()

        self.noise_std = noise_std
        self.device = device

    def forward(self, x):
        noise = torch.normal(0, self.noise_std, size=x.shape).to(self.device)
        return x + noise

## --- END SECTION --- ###    

## --- An Attempt to Create DeepSC --- ##

class DeepSC(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, 
                 tgt_embedding: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer, channel_encoder: ChannelEncoder, 
                 channel_decoder: ChannelDecoder, channel: AWGNChannel):
    # src embedding and tgt embedding are the embedding for the different languages
    # if the same language, embeddings are the same
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.channel_encoder = channel_encoder
        self.channel_decoder = channel_decoder
        self.channel = channel

    # add feature to change channel during training
    def change_channel(self, noise_std, device):
        self.channel = AWGNChannel(noise_std=noise_std, device=device)

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        
        return self.encoder(src, src_mask)
    
    def channel_encode(self, encoder_out):
        return self.channel_encoder(encoder_out)
    
    def channel_transmit(self, in_sig):
        return self.channel(in_sig)
    
    def channel_decode(self, in_sig):
        return self.channel_decoder(in_sig)
    
    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, src, src_mask, tgt, tgt_mask):
        x = self.encode(src, src_mask)
        x = self.channel_encode(x)
        x = self.channel_transmit(x)
        x = self.channel_decode(x)
        x = self.decode(x, src_mask, tgt, tgt_mask)
        x = self.project(x)
        return x


### --- An Attempt to make DeepSC more Modular --- ##


## Encoder + Channel + Channel Decoder
class DeepSC_ECCD(nn.Module):
    def __init__(self, encoder: Encoder, src_embedding: InputEmbeddings, src_pos: PositionalEncoding,
                 channel_encoder: ChannelEncoder, channel: AWGNChannel, channel_decoder: ChannelDecoder):
        super().__init__()
        self.encoder = encoder
        self.src_embedding = src_embedding
        self.src_pos = src_pos
        self.channel_encoder = channel_encoder
        self.channel = channel
        self.channel_decoder = channel_decoder

    def change_channel(self, noise_std, device):
        self.channel = AWGNChannel(noise_std=noise_std, device=device)

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        
        return self.encoder(src, src_mask)
    
    def channel_encode(self, encoder_out):
        return self.channel_encoder(encoder_out)
    
    def channel_transmit(self, in_sig):
        return self.channel(in_sig)
    
    def channel_decode(self, in_sig):
        return self.channel_decoder(in_sig)
    
    def forward(self, src, src_mask):
        x = self.encode(src, src_mask)
        x = self.channel_encode(x)
        x = self.channel_transmit(x)
        x = self.channel_decode(x)
        return x
    
# Decoder, just a typical transformer decoder
class Transformer_Decoder(nn.Module):
    def __init__(self, decoder: Decoder, tgt_embedding: InputEmbeddings, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        
        super().__init__()
        self.decoder = decoder
        self.tgt_embedding = tgt_embedding
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    
    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, src_mask, tgt, tgt_mask):
        x = self.decode(x, src_mask, tgt, tgt_mask)
        x = self.project(x)
        return x
    
# vector input: tgt_vocab_size, each element must be an int
# (I don't want to import list, so we don't check for types hehe)
def Build_MultiDecoder_DeepSC(num_decoders: int, src_vocab_size: int, 
                     tgt_vocab_size, src_seq_len: int, tgt_seq_len: int, device,
                     d_model:int=128, N:int=4, h:int=8, dropout:float=0.1,
                     d_ff:int=512, noise_std:float = 0.1 )->DeepSC:
    
    #create the Input embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)

    #create the positional encoding layers  (They should be the same?)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #create the decoder blocks
    decoder_blocks = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)

        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))


    # create channel related layers
    channel_encoder = ChannelEncoder(d_model, l1_size=256, out_size=16)
    awgn_channel = AWGNChannel(noise_std=noise_std, device=device)
    channel_decoder = ChannelDecoder(in_size=16, l1_size=d_model, l2_size=512)

    # create 1 encoder and channel
    deepsc_encoder_and_channel = DeepSC_ECCD(encoder, src_embed, src_pos, channel_encoder, awgn_channel, channel_decoder)

    # create num_decoders of decoders
    transformer_decoder_blocks = []
    for j in range(num_decoders):
        #create the target embedding layer and the projection layer for the decoders
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size[j])
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size[j])

        transformer_decoder = Transformer_Decoder(decoder, tgt_embed, tgt_pos, projection_layer)
        transformer_decoder_blocks.append(transformer_decoder)

    # Init params
    for p in deepsc_encoder_and_channel.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for i in range(len(transformer_decoder_blocks)):
        for p in transformer_decoder_blocks[i].parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # 1 encoder + channel       an array of decoders
    return deepsc_encoder_and_channel, transformer_decoder_blocks


## --- END Modularity Attempt ---
    
###
''' Big Transfomer parameters
d_model:int=512, N:int=6, h:int=8, dropout:float=0.1,
                     d_ff:int=2048, noise_std:float = 0.1 )->DeepSC:
'''
### Build Simple DeepSC
def Build_DeepSC(src_vocab_size: int, tgt_vocab_size: int,
                     src_seq_len: int, tgt_seq_len: int, device,
                     d_model:int=128, N:int=4, h:int=8, dropout:float=0.1,
                     d_ff:int=512, noise_std:float = 0.1 )->DeepSC:
    
    #create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #create the positional encoding layers  (They should be the same?)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #create the decoder blocks
    decoder_blocks = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)

        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create channel related layers
    channel_encoder = ChannelEncoder(d_model, l1_size=256, out_size=16)
    awgn_channel = AWGNChannel(noise_std=noise_std, device=device)
    channel_decoder = ChannelDecoder(in_size=16, l1_size=d_model, l2_size=512)

    #create the whole transformer
    deepsc = DeepSC(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, 
                    projection_layer, channel_encoder, channel_decoder, awgn_channel)

    # Init params
    for p in deepsc.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return deepsc



# Below is building a normal Transformer, we take some inspiration from this with building the DeepSC

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, 
                 tgt_embedding: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
    # src embedding and tgt embedding are the embedding for the different languages
    # if the same language, embeddings are the same
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)
    
    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def BuildTransformer(src_vocab_size: int, tgt_vocab_size: int,
                     src_seq_len: int, tgt_seq_len: int,
                     d_model:int=512, N:int=6, h:int=8, dropout:float=0.8, d_ff:int=2048)->Transformer:
    
    #create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #create the positional encoding layers  (They should be the same?)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #create the decoder blocks
    decoder_blocks = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)

        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #create the whole transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Init params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer