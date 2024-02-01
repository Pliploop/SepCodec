

## sepcodec class

## forward:
# input: mixed signal, clean signal, class or conditioning vector
# extract embeddings from mixed signal with quantizer or encodec embeddings
# optional (flattening strategy)
# optional (masking strategy for mixture)
# append conditioning vector to embeddings (or add?)
# positional encoding
# transformer encoder encodes mixture
# extract embeddings from clean signal with quantizer or encodec embeddings
# optional (flattening strategy)
# masking strategy for clean signal
# append clean embeddings to mixture embeddings
# if adding conditioning vector, add it here also
# positional encoding
# transformer decoder decodes mixture + clean embeddings
# project to output dimension
# output: estimated clean encodec tokens
# compute loss