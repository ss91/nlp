import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    
    brown_words = []
    brown_tags = []

    entities = []
    temp_words = []
    temp_tags = []
    temp_split = []
    for line in brown_train:
        line = line.strip()
        entities[:] = []
        temp_words[:] = []
        temp_tags[:] = []
        entities = line.split(" ")
        temp_words.append(START_SYMBOL)
        temp_words.append(START_SYMBOL)
        temp_tags.append(START_SYMBOL)
        temp_tags.append(START_SYMBOL)

        for entity in entities:
            temp_split[:] = []
            temp_split = entity.rsplit("/", 1)
            temp_words.append(temp_split[0])
            temp_tags.append(temp_split[1])
        
        temp_words.append(STOP_SYMBOL)
        temp_tags.append(STOP_SYMBOL)
       
        brown_words.append(temp_words[:])
        brown_tags.append(temp_tags[:])
    
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigram_counts = {}
    trigram_counts = {}

    cur_bigram = []
    cur_trigram = []
    
    #print brown_tags

    for tag_list in brown_tags:
        #print tag_list
        for i in range(0, len(tag_list) - 1):
            cur_bigram[:] = []
            
            cur_bigram.append(tag_list[i])
            cur_bigram.append(tag_list[i+1])
            if tuple(cur_bigram) in bigram_counts:
                bigram_counts[tuple(cur_bigram)] = bigram_counts[tuple(cur_bigram)] + 1
            else:
                bigram_counts[tuple(cur_bigram)] = 1
            
            #print cur_bigram
    
        for i in range (0, len(tag_list) - 2):
            cur_trigram[:] = []
        
            cur_trigram.append(tag_list[i])
            cur_trigram.append(tag_list[i+1])
            cur_trigram.append(tag_list[i+2])

            if tuple(cur_trigram) in trigram_counts:
                trigram_counts[tuple(cur_trigram)] = trigram_counts[tuple(cur_trigram)] + 1
            else:
                trigram_counts[tuple(cur_trigram)] = 1
        
            #print cur_trigram
     

    for trigram in trigram_counts:
        #print str(trigram) + " " + str(trigram_counts[trigram])           

 
        if trigram in q_values:
            continue

        cur_q_value = 0
        cur_bigram[:] = []
        cur_bigram.append(trigram[0])
        cur_bigram.append(trigram[1])

        #print trigram
        #print tuple(cur_bigram)

        cur_trigram_prob = float(trigram_counts[trigram])
        cur_bigram_prob = float(bigram_counts[tuple(cur_bigram)])

        cur_q_value = cur_trigram_prob / cur_bigram_prob
        q_values[trigram] = math.log(cur_q_value, 2)
            
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    count = 0
    word_counts = {}
    for sentence in brown_words:
        for word in sentence:
            if word in word_counts:
                word_counts[word] = word_counts[word] + 1
            else:
                word_counts[word] = 1

    for word in word_counts:
        if word_counts[word] > 5:
            count = count + 1
            known_words.add(word)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    
    temp_sentence = []

    
    for sentence in brown_words:
        temp_sentence[:] = []
        for word in sentence:
            if word in known_words:
                temp_sentence.append(word)
            else:
                temp_sentence.append(RARE_SYMBOL)

        brown_words_rare.append(temp_sentence[:])

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    
    tag_word_dict = {}
    local_dict = {}
    for i in range (0, len(brown_words_rare)):
        local_dict = {}
        cur_sentence = brown_words_rare[i]
        cur_tag_seq = brown_tags[i]
        
        for j in range (0, len(cur_tag_seq)):
            cur_tag = cur_tag_seq[j]
            cur_word = cur_sentence[j]

            if cur_tag in tag_word_dict:
                local_dict = tag_word_dict[cur_tag]
                if cur_word in local_dict:
                    local_dict[cur_word] = local_dict[cur_word] + 1
                else:
                    local_dict[cur_word] = 1
            else:
                tag_word_dict[cur_tag] = {}
                local_dict = tag_word_dict[cur_tag]
                local_dict[cur_word] = 1
    
    cur_tag_count = 0 
    cur_word_count = 0
    cur_prob = 0
    cur_tuple = []               
    for tag in tag_word_dict:
        taglist.add(tag)
	cur_tag_count = 0
        local_dict = {}
        local_dict = tag_word_dict[tag]
        for word in local_dict:
            cur_tuple[:] = []
            cur_word_count = 0
            cur_prob = 0
            cur_tag_count = sum(local_dict.values())
            cur_word_count = local_dict[word]
            cur_tuple.append(word)
            cur_tuple.append(tag)
            cur_prob = float (float(cur_word_count)/float(cur_tag_count))
            e_values[tuple(cur_tuple)] = math.log(cur_prob, 2)
            #print str(tag) + " " + str(word) + " " + str(cur_word_count) + " " + str(cur_tag_count) 
    
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!

def possible_tags (taglist, pos):
	possible_tags_new = set([])
	if pos == -1 or pos == 0:
		possible_tags_new.add(START_SYMBOL)
	else:
		possible_tags_new = taglist
	
	return possible_tags_new


def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):

	tagged = []
	start_set = set([])
	start_set.add(START_SYMBOL)
	iter_set_u = set([])
	iter_set_v = taglist
	iter_set_w = set([])
	sentence_mod = []
	cur_pi_list = []
	cur_q_value = 0
	cur_e_value = 0
	sentence_start = 0
	sentence_end = 0
	local_prob = 0
	count = 0

	u_start = 0
	u_end = 0
	v_start = 0
	v_end = 0
	w_start = 0
	w_end = 0


	init_tag_values = []
	def get_backprop_and_pi(list_tuple):
		temp_prob = float(LOG_PROB_OF_ZERO)
		temp_tag = START_SYMBOL

		return max(list_tuple, key=lambda item:item[0]) #referenced from stackoverflow - trying to reduce time


	
	for sentence in brown_dev_words:
		#print sentence
		sentence_start = time.time()

		viterbi_tags = [''] * (len(sentence) + 1)
		sentence_mod = [''] + sentence
		pi = {}
		backprop = {}
		pi[0,'*','*'] = 0
		for k in range (1, len(sentence)+1):
			if k == 1:
				iter_set_u = start_set
			else:
				iter_set_u = taglist
			for u in iter_set_u:
				for v in iter_set_v:
					cur_pi_list[:] = []
					if k == 1 or k == 2:
						iter_set_w = start_set
					else:
						iter_set_w = taglist
					
					for w in iter_set_w:
						local_prob = 0
						
						try:
							cur_q_value = float(q_values[tuple(list((w,u,v)))])
						except KeyError:
							cur_q_value = float(LOG_PROB_OF_ZERO)
							cur_pi_list.append((float(LOG_PROB_OF_ZERO), w))
							continue
						
						if sentence_mod[k] in known_words:
							try:
								cur_e_value = float(e_values[tuple(list((sentence_mod[k],v)))])
							except KeyError:
								cur_e_value = float(LOG_PROB_OF_ZERO)
								cur_pi_list.append((float(LOG_PROB_OF_ZERO), w))
								continue
						else:
							try:
								cur_e_value = float(e_values[tuple(list((RARE_SYMBOL, v)))])
							except KeyError:
								cur_e_value = float(LOG_PROB_OF_ZERO)
								cur_pi_list.append((float(LOG_PROB_OF_ZERO), w))
								continue
					
						local_prob = pi[k-1, w, u] + cur_q_value + cur_e_value
						if local_prob <= float(LOG_PROB_OF_ZERO):
							local_prob = float(LOG_PROB_OF_ZERO)
						
						cur_pi_list.append((float(local_prob),w))

					(pi[k,u,v],backprop[k,u,v]) = get_backprop_and_pi(cur_pi_list)

					
		if len(sentence) == 0:
			iter_set_v = start_set
		else:
			iter_set_v = taglist
		if len(sentence) - 1 <= 0:
			iter_set_u = start_set
		else:
			iter_set_u = taglist

		init_tag_values[:] = []
		
		def get_last_tags(temp_list):
			return max(temp_list, key = lambda item:item[2])
		
		for u in iter_set_u:
			for v in iter_set_v:
				local_prob = 0
				local_prob = float(pi[len(sentence),u,v])
				try:
					local_prob = local_prob + q_values[tuple(list((u, v, STOP_SYMBOL)))]
				except KeyError:
					local_prob = float(LOG_PROB_OF_ZERO)
				
				init_tag_values.append(tuple(list((u,v,float(local_prob)))))		 
					
				
		(viterbi_tags[len(sentence) -1 ], viterbi_tags[len(sentence)], max_prob) = get_last_tags(init_tag_values)

		for k in range (len(sentence)-2, 0, -1):
			viterbi_tags[k] = backprop[k+2, viterbi_tags[k+1], viterbi_tags[k+2]]
		
		tagged_sentence = ''	
		for i in range(0, len(sentence)):
			tagged_sentence = str(tagged_sentence) + str(sentence[i])+"/"+str(viterbi_tags[i+1]) + " "
		
		tagged_sentence.rstrip()
		tagged_sentence = tagged_sentence + "\n"
		tagged.append(tagged_sentence)
	return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
	training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    # IMPLEMENT THE REST OF THE FUNCTION HERE
	tagged = []
	assigned_tag = []
	final_string = ''
	def_tagger = nltk.DefaultTagger('NOUN')
	bigram_tagger = nltk.BigramTagger(training, backoff = def_tagger)
	trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)
	
	#trigram_tagger.evaluate(brown_dev_words)
	#tagger.train(training)
	
	for sentence in brown_dev_words:
		assigned_tag = trigram_tagger.tag(sentence)
		final_string = ''
		for (word,tag) in assigned_tag:
			final_string = final_string + str(word) + "/" + str(tag) + " "

		final_string.rstrip()
		final_string = final_string + "\n"
		tagged.append(final_string)
	return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    #infile = open(DATA_PATH + "test.txt", "r")
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    #infile = open(DATA_PATH + "test.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
