# NEWS-Summary-Sentiment-Analysis
### Checking the Sentiment of the News

![](https://github.com/ShivankUdayawal/NEWS-Summary-Sentiment-Analysis/blob/main/Data%20Visualization/01.jpg)

## Text summarization using machine learning techniques
### A sequence-to-sequence model using an Encoder-Decoder with Attention
The encoder-decoder model for recurrent neural networks is an architecture for sequence-to-sequence prediction problems. It comprised two parts:

 **1. Encoder:** The encoder is responsible for stepping through the input time steps, read the input words one by one and encoding the entire sequence into a fixed length vector called a context vector.

 **2. Decoder:** The decoder is responsible for stepping through the output time steps while reading from the context vector, extracting the words one by one. The trouble with seq2seq is that the only information that the decoder receives from the encoder is the last encoder hidden state which is like a numerical summary of an input sequence. So, for a long input text, we expect the decoder to use just this one vector representation to output a translation. This might lead to catastrophic forgetting.

To solve this problem, the attention mechanism was developed.

**Attention** is proposed as a method to both align and translate. It identifies which parts of the input sequence are relevant to each word in the output (alignment) and use that relevant information to select the right output (translation). So instead of encoding the input sequence into a single fixed context vector (reason for the mentioned bad performance), the attention model develops a context vector that is filtered specifically for each output time step. Attention provides the decoder with information from every encoder hidden state. With this setting, the model can selectively focus on useful parts of the input sequence and hence, learn the alignment between them.

In the next few sections we will go through the whole process: Load the datasets and vector representation, build the vocabulary, define the encoder, decoder and attention mechanism. Then we will code the train stage, iterating over the datasets, and finally we will make the predictions for the validation dataset to get the value of the metrics of interest.

### Content :
The dataset consists of 4515 examples and contains Author_name, Headlines, Url of Article, Short text, Complete Article. I gathered the summarized news from Inshorts and only scraped the news articles from Hindu, Indian times and Guardian. Time period ranges from febrauary to august 2017.

### Data Source :
https://www.kaggle.com/sunnysai12345/news-summary
