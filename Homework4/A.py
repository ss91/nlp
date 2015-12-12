from __future__ import print_function
import nltk
from nltk.corpus import comtrans
from nltk.align import IBMModel1, IBMModel2, AlignedSent, Alignment
# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):

    en_ibm1 = IBMModel1(aligned_sents, 10)
    return en_ibm1    

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    
    en_ibm2 = IBMModel2(aligned_sents, 10)
    return en_ibm2

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    
    total_aer = float(0)
    for x in range(0,n):
        item = aligned_sents[x]
        temp = model.align(item)
        total_aer = total_aer + float(temp.alignment_error_rate(item))
        
    avg_aer = float(total_aer)/float(n)
    return avg_aer

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    
    f = open(file_name, 'w')
    
    for x in range (0,20):
        item = aligned_sents[x]
        temp = model.align(item)
        print(temp.mots, file = f)
        print(temp.words, file = f)
        print (temp.alignment, file = f)
        f.write('\n')
    
    f.close()
         
    return
def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
