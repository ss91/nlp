NLP - Homework4
Sankalp Singayapally - ss4728

(varies with load on clic machines)
Total Running Time - Part A and B is:

real	4m43.572s
user	4m40.835s
sys	0m2.733s

Part A

Used a simple averaging scheme here (arithmetic mean of all the reported AERs).

1. Works as expected
2. Works as expected

3. Average AER for IBM Model 1 - 0.665, for IBM Model 2 - 0.650

[u'I', u'would', u'like', u'your', u'advice', u'about', u'Rule', u'143',
u'concerning', u'inadmissibility', u'.']
[u'K\xf6nnten', u'Sie', u'mir', u'eine', u'Auskunft', u'zu', u'Artikel', u'143',
u'im', u'Zusammenhang', u'mit', u'der', u'Unzul\xe4ssigkeit', u'geben', u'?']

This sentence has a considerable difference in the AER between the 2 models.
Model 1 scores it at 0.714 and Model 2 at 0.785. This is among the few sentences
for which Model 1 seems to perform better than Model 2. I believe that, while in
general model2 is better, in this particular case, model 1 performs better
because the sentence doesn't have many multiple mappings i.e. there doesn't seem
to be much ambiguity in the alignments. This is good for model1, which does only
lexical translation. However, in the majority of the other cases, the reordering
model of model2 helps it achieve better performance. When the reordering is
done, it accounts for the distance between the word in the source sentence and
the aligned word in the destination sentence and factoring that in will help it
make a better decision about the final alignment. 

On the other hand, an example where model2 performs better is:

[u'Dennoch', u',', u'Frau', u'Pr\xe4sidentin', u',', u'wurde', u'meinem', u'Wunsch', u'nicht', u'entsprochen', u'.']
[u'But', u',', u'Madam', u'President', u',', u'my', u'personal', u'request', u'has', u'not', u'been', u'met', u'.']

Model2 has an AER of 0.416666666667 and Model1 has an AER of 0.652173913043

This is because in addition to the translation, the position appears to be critical, something that Model 2 accounts for. On the other hand, in the previous sentence, given the mappings, the translation is the critical part and hence, model1 was performing better. When one looks at the alignments within this sentence, it is evident that there is some positional distortion and that is why model2 performs better. 

4. Iterations to convergence

Some local optimum appears to be at an AER of 0.661 for Model 1 and 0.648 for Model 2. 
This is the value for 20 iterations and at 100 iterations, Model 2 has a higher
AER than before implying that with a larger number of iterations, there are more
errors. It's possible that with a large number of iterations, the data is
getting overtrained or getting stuck at a local minimum, but since we are testing on the same set, I am not sure I
can explain why the error rate is higher! For 5 iterations, AER of Model 1 is 0.627 and for 4 iterations for Model 2 is 0.642 which seems like a better convergent solution. I think
this is closer to being a global minimum. I think since we have a small dataset, fewer iterations give us a satisfactory minimum.
With a large dataset, we may need to run the entire thing for longer to get a more convergent and solid result. 


Part B

*I pretty much ripped off the implementation from NLTK IBMModel2 (including variable names :P) and it looks a little confusing. Basically, I consider NLTK to be doing forward translation and I do the same steps for "backward" as well and average it out. 


I was unable to reach the reference AER but came close to it and the results are here:

Berkeley Aligner
---------------------------
Average AER: 0.576


A sentence for which Berkeley aligner outperforms both the others is:

0.166666666667
[u'Dennoch', u',', u'Frau', u'Pr\xe4sidentin', u',', u'wurde', u'meinem', u'Wunsch', u'nicht', u'entsprochen', u'.']
[u'But', u',', u'Madam', u'President', u',', u'my', u'personal', u'request', u'has', u'not', u'been', u'met', u'.']

For this sentence: BA > I2 > I1 and significantly at that. Again, the explanation here would be that the berkeley aligner, in addition to accounting for position distortion, does it both ways and the average will give the "best of both worlds" scenario, I suppose. Looking at some other sentences, I believe my reasoning is right. I have used a simple averaging scheme here but I think we can get more intuitive results if we adjust the weights.