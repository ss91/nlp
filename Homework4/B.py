import nltk
import A
from nltk.align import AlignedSent, IBMModel1
from collections import defaultdict

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):

        self.t_f = defaultdict(lambda: defaultdict(float))
        self.t_b = defaultdict(lambda: defaultdict(float))
        self.q_f = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
        self.q_b = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

        #initalization step

        self.all_words = set()
        self.all_mots = set()

        for sentence in align_sents:
            num_mots = len(sentence.mots)
            num_words = len(sentence.words)

            self.all_words.update(sentence.words)
            self.all_mots.update(sentence.mots)

            initial_value_mots = float(1)/(num_mots + 1)
            initial_value_words = float(1)/(num_words + 1)

            for i in range (0, num_mots + 1):
                for j in range (1, num_words + 1):
                    self.q_f[i][j][num_mots][num_words] = initial_value_mots

            for i in range (0, num_words + 1):
                for j in range (1, num_mots + 1):
                    self.q_b[i][j][num_words][num_mots] = initial_value_mots

        ibm1 = IBMModel1(align_sents, 10)
        self.t_f = ibm1.probabilities



        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):

        best_alignment = []
        cur_words = align_sent.words
        cur_mots = align_sent.mots
        cur_words_len = len(cur_words)
        cur_mots_len = len(cur_mots)

        for j, trg_word in enumerate(cur_words):
            best_prob = self.t[trg_word][None]*self.q[0][j+1][cur_mots_len][cur_words_len]
            best_alignment_point = None
            for i, src_word in enumerate(cur_mots):
                align_prob = self.t[trg_word][src_word] * self.q[i+1][j+1][cur_mots_len][cur_words_len]
                
                if align_prob >= best_prob:
                    best_prob = align_prob
                    best_alignment_point = i

            if best_alignment_point is not None:
                best_alignment.append((j, best_alignment_point))    

        return AlignedSent(cur_words, cur_mots, best_alignment)
    
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):

        # lets try with the pseudocode and the general outline that was posted on Piazza
        # okay, I was thinking we had to do two IBMModel2s and then play around with their probabilities
        # but turns out that isn't as easy as I thought.

        #FORWARD is what IBM2 does - Backward is the extra we are doing
        #FORWARD translation is WORD,MOT
        #FORWARD distortion is MOT,WORD

        t_f = self.t_f 
        t_b = self.t_b 
        q_f = self.q_f 
        q_b = self.q_b 
        all_words = self.all_words #set()
        all_mots = self.all_mots #set()

        #begin EM iterations
        #okay, pretty much all of this is ripped off IBM2 - just add more loops to reverse
        #and then do the averaging

        for i in range (0, num_iters):
            count_t_given_s_f = defaultdict(lambda: defaultdict(float))
            count_any_t_given_s_f = defaultdict(float)
            count_t_given_s_b = defaultdict(lambda: defaultdict(float))
            count_any_t_given_s_b = defaultdict(float)
            alignment_count_f = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            alignment_count_for_any_i_f = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
            alignment_count_b = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            alignment_count_for_any_i_b = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

            for sentence in aligned_sents:
                cur_mots = [None] + sentence.mots
                cur_words = ['UNUSED'] + sentence.words

                cur_mots_len = len(sentence.mots)
                cur_words_len = len(sentence.words)

                total_count_f = defaultdict(float)
                total_count_b = defaultdict(float)
                for j in range(1, cur_words_len + 1):
                    t = cur_words[j]
                    total_count_f[t] = 0

                    for i in range (0, cur_mots_len + 1):
                        s = cur_mots[i]
                        count = t_f[t][s] * q_f[i][j][cur_mots_len][cur_words_len]

                        #print (t_f[t][s], q_f[i][j][cur_mots_len][cur_words_len])

                        total_count_f[t] += count

                for j in range(1, cur_words_len + 1):
                    t = cur_words[j]

                    for i in range (0, cur_mots_len + 1):
                        s = cur_mots[i]
                        count = t_f[t][s] * q_f[i][j][cur_mots_len][cur_words_len]
                        #print total_count[j]
                        normalized_count = count / total_count_f[t]

                        count_t_given_s_f[t][s] += normalized_count
                        count_any_t_given_s_f[s] += normalized_count
                        alignment_count_f[i][j][cur_mots_len][cur_words_len] += normalized_count
                        alignment_count_for_any_i_f[j][cur_mots_len][cur_words_len] += normalized_count

            
                cur_words = [None] + sentence.words
                cur_mots = ['UNUSED'] + sentence.mots

                for j in range(1, cur_mots_len + 1):
                    s = cur_mots[j]
                    total_count_b[s] = 0

                    for i in range(0, cur_words_len + 1):
                        t = cur_words[i]
                        count = t_f[s][t] * q_b[i][j][cur_words_len][cur_mots_len]
                        total_count_b[s] += count

                for j in range (1, cur_mots_len + 1):
                    s = cur_mots[j]

                    for i in range(0, cur_words_len + 1):
                        t = cur_words[i]
                        count = t_f[s][t] * q_b[i][j][cur_words_len][cur_mots_len]

                        normalized_count = count / total_count_b[s]

                        count_t_given_s_b[s][t] += normalized_count
                        count_any_t_given_s_b[t] += normalized_count
                        alignment_count_b[i][j][cur_words_len][cur_mots_len] += normalized_count
                        alignment_count_for_any_i_b[j][cur_words_len][cur_mots_len] += normalized_count
           

            for t in all_words:
                for s in all_mots:

                    cur_probability = (count_t_given_s_f[t][s]/count_any_t_given_s_f[s] + count_t_given_s_b[s][t]/count_any_t_given_s_b[t]) / 2
                    t_f[s][t] = cur_probability
                    t_f[t][s] = cur_probability


            for sentence in aligned_sents:
                cur_mots_len = len(sentence.mots)
                cur_words_len = len(sentence.words)

                for i in range(0, cur_mots_len + 1):
                    for j in range (1, cur_words_len + 1):
                        try:
                            cur_probability = alignment_count_f[i][j][cur_mots_len][cur_words_len] / alignment_count_for_any_i_f[j][cur_mots_len][cur_words_len] + alignment_count_b[j][i][cur_words_len][cur_mots_len] / alignment_count_for_any_i_b[i][cur_words_len][cur_mots_len]
                            q_f[i][j][cur_mots_len][cur_words_len] = cur_probability / 2 
                        except ZeroDivisionError:
                            q_f[i][j][cur_mots_len][cur_words_len] = alignment_count_f[i][j][cur_mots_len][cur_words_len] / alignment_count_for_any_i_f[j][cur_mots_len][cur_words_len]     

        return (t_f, q_f)                            



def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
