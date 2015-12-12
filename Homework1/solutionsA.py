import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000
# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    
    word_counts={}
    bigram_counts={}
    trigram_counts={}
    cur_tokens = []
    temp = []
    tmp_bigram=[]
    unigrams=[]
    bigrams=[]
    trigrams=[] 

    for line in training_corpus:
        cur_tokens[:] = []
        #line = line.strip('\n')
        line = line.strip()
        cur_tokens.append(str(START_SYMBOL))
        cur_tokens.append(str(START_SYMBOL))
        for item in line.split(" "):
            cur_tokens.append(str(item))
        cur_tokens.append(str(STOP_SYMBOL))
        for token in cur_tokens:
            if str(token) == str(START_SYMBOL):
                continue
            else:
                unigrams.append(tuple([token]))
        
        for i in range(1, len(cur_tokens)-1):
            temp[:] = []
            temp.append(cur_tokens[i])
            temp.append(cur_tokens[i+1])
            bigrams.append(tuple(temp))
            
        for i in range (0, len(cur_tokens) - 2):
            temp[:] = []
            temp.append(cur_tokens[i])
            temp.append(cur_tokens[i+1])
            temp.append(cur_tokens[i+2])
            trigrams.append(tuple(temp))         
  
    
    for unigram in unigrams:
        if unigram in word_counts:
            word_counts[unigram] = word_counts[unigram] + 1
        else:
            word_counts[unigram] = 1
    
    for bigram in bigrams:
        if bigram in bigram_counts:
            bigram_counts[bigram] = bigram_counts[bigram] + 1
        else:
            bigram_counts[bigram] = 1

    for trigram in trigrams:
        if trigram in trigram_counts:
            trigram_counts[trigram] = trigram_counts[trigram] + 1
        else:
            trigram_counts[trigram] = 1

    unigram_total = float(sum(word_counts.itervalues()))
    for unigram in unigrams:
        count = 0
        count = float(word_counts[unigram])
        unigram_p[unigram] = math.log(count/unigram_total,2) 
    
    for bigram in bigrams:
        count = 0
        count = float(bigram_counts[bigram])
        try:
            bigram_p[bigram] = math.log( (count / word_counts[tuple([str(bigram[0])])]), 2)
        except KeyError:
            bigram_p[bigram] = math.log( (count / word_counts[tuple([STOP_SYMBOL])]), 2)
    
    for trigram in trigrams:
        tmp_bigram[:] = []
        count = 0
        count = float(trigram_counts[trigram])
        tmp_bigram.append(trigram[0])
        tmp_bigram.append(trigram[1])
        try:
            trigram_p[trigram] = math.log( (count / bigram_counts[tuple(tmp_bigram)]), 2)
        except KeyError:
            if tmp_bigram[0] == '*' and tmp_bigram[1] == '*':
                trigram_p[trigram] = math.log( (count / word_counts[tuple([STOP_SYMBOL])]), 2)
            else:
                trigram_p[trigram] = MINUS_INFINITY_SENTENCE_LOG_PROB
        
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    
    cur_tokens = []    
    scores = []
    cur_score = 0
    temp = []
    for line in corpus:
        cur_score = 0
        cur_tokens[:] = []
        cur_tokens.append(str(START_SYMBOL))
        cur_tokens.append(str(START_SYMBOL))
        line = line.strip()
        for item in line.split(" "):
            cur_tokens.append(item)
        cur_tokens.append(str(STOP_SYMBOL))
        if n == 1:
            #cur_score = 0
            #calculating scores based on unigrams now
            for token in cur_tokens:
                if token == START_SYMBOL:
                    continue
                cur_score = cur_score + float(ngram_p[tuple([token])]) #since we already have log probabilites might as well just add them instead of using exponent operator here
   
        if n == 2:
            #calculate scores based on bigrams now
            for i in range (1, len(cur_tokens) -1):
                temp[:] = []
                temp.append(cur_tokens[i])
                temp.append(cur_tokens[i+1])
                cur_score = cur_score + float(ngram_p[tuple(temp)])
            
        
        if n == 3:
            for i in range (0, len(cur_tokens) - 2):
                temp[:] = []
                temp.append(cur_tokens[i])
                temp.append(cur_tokens[i+1])
                temp.append(cur_tokens[i+2])
                cur_score = cur_score + float(ngram_p[tuple(temp)])
 
        scores.append(cur_score)
    
    #print scores
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []

    cur_tokens = []
    temp_unigram = []
    temp_bigram = []
    temp_trigram = []
    cur_unigram_p = 0
    cur_bigram_p = 0
    cur_trigram_p = 0
    neg_uni_inf_flag = 0
    neg_bi_inf_flag = 0
    neg_tri_inf_flag = 0
    neg_inf_flag = 0
    cur_linear_score = 0
    for line in corpus:
        cur_linear_score = 0
        neg_uni_inf_flag = 0
        neg_bi_inf_flag = 0
        neg_tri_inf_flag = 0
        neg_inf_flag = 0
        cur_tokens[:] = []
        cur_tokens.append(str(START_SYMBOL))
        cur_tokens.append(str(START_SYMBOL))
        line = line.strip()
        for item in line.split(" "):
            cur_tokens.append(item)
        cur_tokens.append(STOP_SYMBOL)

        for i in range (0, len(cur_tokens)-2):
            neg_inf_flag = 0
            trigram_linear_score = 0
            temp_unigram[:] = []
            temp_bigram[:] = []
            temp_trigram[:] = []
            temp_trigram.append(cur_tokens[i])
            temp_trigram.append(cur_tokens[i+1])
            temp_trigram.append(cur_tokens[i+2])

            temp_unigram.append(temp_trigram[2])
            temp_bigram.append(temp_trigram[1])
            temp_bigram.append(temp_trigram[2])
                
            #print temp_unigram
            #print temp_bigram
            #print temp_trigram

            try:
                cur_unigram_p = float(unigrams[tuple(temp_unigram)])
            except KeyError:
                #print cur_unigram_p
                if str(temp_unigram[0]) == str(START_SYMBOL):
                    continue
                else:
                    cur_unigram_p = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    neg_inf_flag = 1

            try:
                cur_bigram_p = float(bigrams[tuple(temp_bigram)])
                #print cur_bigram_p
            except KeyError:
                #print temp_bigram
                if ((str(temp_bigram[0]) == str(START_SYMBOL)) and (str(temp_bigram[1]) == str(START_SYMBOL))):
                    continue
                else:
                    cur_bigram_p = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    neg_inf_flag = 1
            try:
                cur_trigram_p = float(trigrams[tuple(temp_trigram)])
                #print temp_trigram
                #print cur_trigram_p
            except KeyError:
                cur_trigram_p = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
                neg_inf_flag = 1
            
            #if ((cur_unigram_p == float(MINUS_INFINITY_SENTENCE_LOG_PROB)) or (cur_bigram_p == float(MINUS_INFINITY_SENTENCE_LOG_PROB)) or (cur_trigram_p == float(MINUS_INFINITY_SENTENCE_LOG_PROB))):
            #    cur_score = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
            
            if neg_inf_flag == 1:
			    #cur_linear_score = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
				if cur_bigram_p == float(MINUS_INFINITY_SENTENCE_LOG_PROB):
					if cur_unigram_p == float(MINUS_INFINITY_SENTENCE_LOG_PROB):
						cur_linear_score = float(MINUS_INFINITY_SENTENCE_LOG_PROB)
						break
					else:
						trigram_linear_score = math.log(math.pow(2,cur_unigram_p)/3,2)
				else:
						trigram_linear_score = math.log((math.pow(2, cur_unigram_p) + math.pow(2,cur_bigram_p))/3, 2)
			
            else:
                #print str(temp_unigram) + " " + str(cur_unigram_p)
                #print str(temp_bigram) + " " + str(cur_bigram_p)
                #print str(temp_trigram) + " " + str(cur_trigram_p)
                trigram_linear_score = math.log(((math.pow(2, cur_unigram_p)) + (math.pow(2, cur_bigram_p)) + (math.pow(2, cur_trigram_p)))/3 , 2)
                #print trigram_linear_score 
            cur_linear_score = cur_linear_score + trigram_linear_score

        scores.append(float(cur_linear_score))
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
