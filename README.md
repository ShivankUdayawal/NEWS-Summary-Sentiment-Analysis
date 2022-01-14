# NEWS-Summary-Sentiment-Analysis
### Checking the Sentiment of the News

## Text summarization using machine learning techniques
### A sequence-to-sequence model using an Encoder-Decoder with Attention
The encoder-decoder model for recurrent neural networks is an architecture for sequence-to-sequence prediction problems. It comprised two parts:

 **1. Encoder:** The encoder is responsible for stepping through the input time steps, read the input words one by one and encoding the entire sequence into a fixed length vector called a context vector.

 **2. Decoder:** The decoder is responsible for stepping through the output time steps while reading from the context vector, extracting the words one by one. The trouble with seq2seq is that the only information that the decoder receives from the encoder is the last encoder hidden state which is like a numerical summary of an input sequence. So, for a long input text, we expect the decoder to use just this one vector representation to output a translation. This might lead to catastrophic forgetting.

To solve this problem, the attention mechanism was developed.
