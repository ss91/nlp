Sankalp Singayapally - ss4728

Part A

1. 

UNIGRAM natural -13.766408817
BIGRAM natural that -4.05889368905
TRIGRAM natural that he -1.58496250072

2.

The perplexity is 1052.4865859 - unigram model
The perplexity is 53.8984761198 - bigram model
The perplexity is 5.7106793082 - trigram model

3.

The perplexity with the linear interpolation model is 12.5516094886. 

4.

The linear interpolation perplexity is better than both the unigram and bigram models but not the trigram models. The reason is because we are generating the model using a corpus and calculating the perplexity on the same corpus, the trigram model is the best one we can get since among the three, trigrams are the more accurate ones since they have more context. When we use linear interpolation, we are using the less accurate models here by giving them some weight (in this case equal weights) and thereby reducing the accuracy of the model which increases the perplexity.

5.

The perplexity is 11.1670289158 - Sample 1
The perplexity is 1611240282.44 - Sample 2

Based on the perplexities, we can say that Sample 1 is similar to the Brown corpus training set. Broadly, a language model that produces lower perplexity is "better". In this case, since the model is trained on a dataset, it is natural that it can better predict the same one because of a bias. A lower perplexity indicates that it is indeed able to do that. In fact, when we compare the two sets, we see that sample2 has no lines at all in common with the original training set and in fact has some sentences in a different language which would explain why there are a large number of missing unigrams

Part B

2.

TRIGRAM CONJ ADV ADP -2.9755173148
TRIGRAM DET NOUN NUM -8.9700526163
TRIGRAM NOUN PRT PRON -11.0854724592

3.

First two lines of output matches the sample given in the assignment

4.

* * 0.0
Night NOUN -13.8819025994
Place VERB -15.4538814891
prime ADJ -10.6948327183
STOP STOP 0.0
_RARE_ VERB -3.17732085089

5.

Percent correct tags: 93.1897512068
This is a little off from the check given in the assignment. I think it's because of the way I handle some exceptions when assigning tags in the first backpropagation step. Had to do this for 1 or 2 specific cases - else most of the output matches 

6.
Percent correct tags: 87.9985146677
Note - if we create a unigram tagger and have the bigram tagger backoff to the unigram tagger and the unigram to the default one, then the accuracy improves to 94%. 

Execution times:
Part A time: 18.04 sec
Part B time: 1815.98 sec - I know this is very slow but I am at a loss of how to optimize it. 
