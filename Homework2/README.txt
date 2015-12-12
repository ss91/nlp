COMS W4705 - Natural Language Processing
Assignment II
Sankalp Singayapally - ss4728

Part 1:

a - All four png files are provided
b - A dependency graph is projective if it doesn't have any dependency arcs crossing each other. In the code provided in the file providedcode/transitionparser.py the function _is_projective checks if the dependency graph is projective or not. Given any two nodes, parent and child, the function checks if there is already an arc between some node k that lies between the parent and child nodes and any other node in the graph. If such an arc exists then it means that it will cross over the arc between parent and child - a very intuitive way to check if a graph is projective. Since every pair of parent-child nodes are checked, we can be sure that if a graph passes this test, it is projective.

c - English sentence with projective dependency graph: 

    I saw him walk to college.

    English sentence without a projective dependency graph:

    I have to write an exam today, on NLP (NLP is dependent on exam and today is dependent on write so this not projective)
      

Part 2:

The missing operations have been implemented according to the algorithm, including the preconditions. Using the badfeatures.model that was provided, the performance i.e. the UAS and LAS are pretty low (as expected). The features we are using to train here are pretty basic. We just use feats and word for the top of the stack and the buffer and calculate their dependencies which is intuitive to some extent but definitely not sufficient. I believe a major reason that these features do not give us accurate predictions is because we are not accounting for ambiguity, which can be handled by using features such as POSTAG. 


UAS: 0.231228838877 
LAS: 0.125473013344

Part 3:

a - Discussion of features:

I've chosen to add the following features to train the classifier (in addition to what were originally given)

STK[0] - POSTAG
STK[0] - LEMMA
STK[1] - POSTAG
BUF[0] - POSTAG
BUF[0] - LEMMA
BUF[1] - FORM
BUF[1] - POSTAG

I believe adding the POSTAG features provided the most significant increase in accuracy as this provides a much needed "context" to the classifier. By this, I mean that using the tag, some degree of word sense ambiguity can be eliminated - for example, different usages of the word "run" as a noun or a verb. 

I've chosen to use the POSTAG feature for the top two words in the stack and the buffer because, as per my understanding, these are the ones that are most commonly seen in a configuration and are used to decide the transition. The more information that is provided about these words, the more accurate the classifer's decision will be. 

I've also gone ahead and added the LEMMA for the top of the stack and the buffer. Again, this is something that will provide the classifier with a more robust input and it won't get overtrained between a word and its plural (This is an argument based on my intuition and I am not sure if I am completely right here). For example, if the classifier had seen only the word "rule" so far and it now comes across "rules", using the LEMMA feature will help it make a better decision there as it is similar to a feature that it has already seen. 

The justification for using FORM for BUF[1] is similar to that of using POSTAG - it just provides better context since BUF[1] is rather common among configurations. 

Though not in the submission, I want to note that I did use CTAG as well while experimenting and quite honestly, I think this is not a very useful feature. It in fact might provide ambiguous information to the classifier since we are already using POSTAG. However, if we use CTAG for words deeper in the buffer, BUF[3] etc., we might see some improvement in the score because it can provide the classifier a better sense of the overall context.

I've followed the implementation provided in the sample to implement all of the features I added. However, I suppose there is no possibility of multiple tags and lemmas so the loops there would be redundant. I think all three variants of the features I've chosen, POSTAG, LEMMA and FORM have a constant time complexity. However, by adding BUF[1] and STK[1] features to the mix, I would have increased the complexity of the overall system by a small constant amount. I don't think this is linear time complexity because for each configuration, the increase is fixed. 

b - Implemented the features

c - Scoring the models for Swedish, Danish and English

Swedish:
UAS: 0.7759410476 
LAS: 0.667795259908

Danish:
UAS: 0.786826347305 
LAS: 0.713373253493

English: 

Using dev corpus:
UAS: 0.720987654321 
LAS: 0.683950617284

Using test corpus:
UAS: 0.795597484277 
LAS: 0.738993710692

d - Limitations of the arc-eager parser

The arc-eager parser is deterministic in the sense that it will give us a transition but the transition itself is probabilistic since it is generated through a standard ML concept. As with any trained system, the accuracy of this will only be as good as the training data and the training parameters. We've seen in this assignment itself that it is heavily dependent on the features provided to it. We're lucky to have a dataset that can be used to train the model with a range of features because otherwise the model would be pretty poor. The arc-eager parser has a linear time complexity depending on the length of the sentence. Like I mentioned before, this parser makes a trade-off between accuracy and efficiency since, at its heart it is not fully deterministic but since it gives us reasonably good results with an impressive time complexity, it is perhaps one of the best ones to use. 

Part 4:

Works as expected
parse.py prints to stdout which is redirected as per the specification
