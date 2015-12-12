import A
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import stopwords
from sklearn import svm
from sklearn import neighbors
import nltk
import string
import heapq
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from nltk.stem.snowball import SnowballStemmer
from nltk.data import load
from nltk.corpus import wordnet as wn
unigram_tagger = None
bigram_tagger = None 
trigram_tagger = None
stemmer = None
# You might change the window size
window_size = 3
global_language = ""
punctuation_unicode = []
stop_words = [] #make these global to save on time and call them only once
# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''

    
    features = {}
    labels = {}

    sense_words_count = {}
    words_sense_count = {}
    #let's start with the simple thing shall we?
    #just keep the counts of the words within the window size like we did in part A

    set_1 = set()
    for item in data:
        left_context_tokens = nltk.word_tokenize(item[1])
        set_1 = set_1.union(set(left_context_tokens))
        right_context_tokens = nltk.word_tokenize(item[3])
        set_1 = set_1.union(set(right_context_tokens))


    
   
    #let's try the relevance score thing once again - only for english since the other two are getting the reference scores
    #basically get the top words for each instance and use the counts of the words within a window as features?
   
    
    for instance in data:
        temp_dict = {}
        cur_instance_id = instance[0]
        cur_instance_left_context = nltk.word_tokenize(instance[1])
        cur_instance_right_context = nltk.word_tokenize(instance[3])
        cur_instance_sense_id = instance[4]

        if cur_instance_sense_id not in sense_words_count.keys():
            sense_words_count[cur_instance_sense_id] = {}
       
        
        cur_instance_left_context_filtered = [word for word in cur_instance_left_context if word not in stop_words]
        cur_instance_right_context_filtered = [word for word in cur_instance_right_context if word not in stop_words]

        '''
        if global_language == "english":

            for word in cur_instance_left_context_filtered:
                word_s = stemmer.stem(word)
                if word_s in sense_words_count[cur_instance_sense_id]:
                    sense_words_count[cur_instance_sense_id][word_s] = sense_words_count[cur_instance_sense_id][word_s] + 1
                else:
                    sense_words_count[cur_instance_sense_id][word_s] = 1
                    
                if word_s not in words_sense_count.keys():
                    words_sense_count[word_s] = {}
                else:
                    if cur_instance_sense_id not in words_sense_count[word_s].keys():
                        words_sense_count[word_s][cur_instance_sense_id] = 1
                    else:
                        words_sense_count[word_s][cur_instance_sense_id] = words_sense_count[word_s][cur_instance_sense_id] + 1
                    
            for word in cur_instance_right_context_filtered:
                word_s = stemmer.stem(word)
                if word_s in sense_words_count[cur_instance_sense_id]:
                    sense_words_count[cur_instance_sense_id][word_s] = sense_words_count[cur_instance_sense_id][word_s] + 1
                else:
                    sense_words_count[cur_instance_sense_id][word_s] = 1   
            
                if word_s not in words_sense_count.keys():
                    words_sense_count[word_s] = {}
                else:
                    if cur_instance_sense_id not in words_sense_count[word_s].keys():
                        words_sense_count[word_s][cur_instance_sense_id] = 1
                    else:
                        words_sense_count[word_s][cur_instance_sense_id] = words_sense_count[word_s][cur_instance_sense_id] + 1

        '''    
        
        cur_instance_left_context_filtered_f = [word for word in cur_instance_left_context if word not in punctuation_unicode] #has stop words    
        cur_instance_right_context_filtered_f = [word for word in cur_instance_right_context if word not in punctuation_unicode] #has stop words     
        cur_instance_left_context_windowed = cur_instance_left_context_filtered_f[-window_size:] 
        cur_instance_right_context_windowed = cur_instance_right_context_filtered_f[:window_size]

        word_count = 0
        for word in set_1:
            word_count = 0
            word_count = word_count + cur_instance_left_context_filtered.count(word)
            word_count = word_count + cur_instance_right_context_filtered.count(word)
            temp_dict[word] = word_count
        
        temp_dict['head'] = stemmer.stem(item[2])
        head_tag = trigram_tagger.tag((item[2],))[0][1]
       
        if global_language == "english":
            synonym_set = set()
            for ss in wn.synsets(item[2]):
                synonym_set.update(ss.lemma_names())

            syn_list = list(synonym_set)
            for i in range (0, len(syn_list)):
                count = 0
                count = count + cur_instance_left_context_filtered.count(syn_list[i])
                count = count + cur_instance_right_context_filtered.count(syn_list[i])
                temp_dict['syn'+str(i)] = count
       
        #okay we have added the counts of the synonyms for head into the feature now
       
        
        if head_tag != None:    
            temp_dict['pos0'] = head_tag 
        
        for i in range (0, window_size):
            try:
                temp_dict['w-' + str(i)] = stemmer.stem(cur_instance_left_context_windowed[i])
            except IndexError:
                continue
    
        
        for i in range (0, window_size):
            try:
                temp_dict['w' + str(i)] = stemmer.stem(cur_instance_right_context_windowed[i])
            except IndexError:
                continue

        for x in range (0, window_size):
            cur_feature = 'pos-' + str(x+1)
            try:
                tag = trigram_tagger.tag((cur_instance_left_context_windowed[x],))[0][1]
                if tag != None:
                    temp_dict[cur_feature] = tag
            except IndexError:
                continue
        
        
        for x in range (0, window_size):
            cur_feature = 'pos' + str(x+1)
            try:
                tag = trigram_tagger.tag((cur_instance_right_context_windowed[x],))[0][1]
                if tag != None:
                    temp_dict[cur_feature] = tag
            except IndexError:
                continue
        
        
        features[cur_instance_id] = temp_dict
        labels[cur_instance_id] = instance[4]

    
    '''
    we have this:
    sense_words_count = {sense_1: {w1: count, w2: count....
                        sense_2: {w1: count, w2: count....
                        ...

    words_sense_count = {w1: {sense_1: count, sense_2: count....
                         w2: {sense_1: count, sense_2: count....
                        
    '''

    #this is not really helping and taking more time than I'd like
    #I'm getting the top words and adding their counts in the contexts to the feature vectors

    ''' 
    if global_language == "english":

        relevance_score_dict = {}
        top_words_dict = {}

        for sense_id in sense_words_count.keys():
            if sense_id not in relevance_score_dict.keys():
                relevance_score_dict[sense_id] = {}

            for word in sense_words_count[sense_id].keys():
                cur_count = sense_words_count[sense_id][word]
                total_count = sum(words_sense_count[word].values())

                try:
                    rel_score = float(cur_count/(total_count - cur_count ))
                except ZeroDivisionError:
                    rel_score = -10000000
                
                relevance_score_dict[sense_id][word] = rel_score    
        
        
        for sense_id in relevance_score_dict.keys():
            k_keys_sorted = heapq.nlargest(10, relevance_score_dict[sense_id], key = relevance_score_dict[sense_id].get)
            top_words_dict[sense_id] = k_keys_sorted

        
        
        for instance in data:
            cur_instance_id = instance[0]
            cur_feature_dict = features[cur_instance_id]
            cur_sense_id = instance[4]

            cur_left_tokens = nltk.word_tokenize(instance[1])
            cur_right_tokens = nltk.word_tokenize(instance[3])

            cur_top_words = top_words_dict[cur_sense_id]
            
            for i in range (0, len(cur_top_words)):
                cur_top_word_count = 0
                cur_top_word_count = cur_left_tokens.count(cur_top_words[i])
                cur_top_word_count = cur_top_word_count + cur_right_tokens.count(cur_top_words[i])

                features[cur_instance_id]['top'+str(i)] = cur_top_word_count
        
        '''

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    
    return X_train, X_test
    
    
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''
    '''
    temp_new = []
    label_new = []
    temp_new_test = []
    new_train = {}
    new_test = {}
    train_keys = X_train.keys()
    test_keys = X_test.keys()
    for instance_id in train_keys:
        temp_new.append(X_train[instance_id])
        label_new.append(y_train[instance_id])
        
    for instance_id in test_keys:
        temp_new_test.append(X_test[instance_id])

    X_train_new = SelectKBest(chi2, k=10)

    
    for i in range (0, len(temp_new)):
        train_fit = X_train_new.transform(temp_new[i])
        train_f[train_keys[i]] = train_fit

    for i in range(0, len(test_keys)):
        test_fit = X_train_new.transform(temp_new_test[i])
        test_f[test_keys[i]] = test_fit

    return train_fit, test_fit
    '''

    '''
    train_fit = X_train_new.fit_transform(temp_new, label_new)
    test_fit = X_train_new.transform(temp_new_test)
    train_f = {}
    test_f = {}
    for i in range (0, len(train_keys)):
        train_f[train_keys[i]] = train_fit[i]

    for i in range (0, len(test_keys)):
        test_f[test_keys[i]] = test_fit[i]

        return train_f, test_f
    '''
    #saving the keys in a list to ensure that the order is maintained
    #assuming that the classifier doesn't mess around with that
    #anyway, this is reducing the performance
    
    #return X_train_new, X_test_new  


    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    #return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''
    results = []
    svm_results = []
    knn_results = []
    svm_clf = svm.LinearSVC()
    #knn_clf = neighbors.KNeighborsClassifier()

    train_formatted = []
    label_formatted = []
    test_formatted = []
   
    for item in X_train:
        train_formatted.append(X_train[item])
        label_formatted.append(y_train[item])

    #try:
    #knn_clf.fit(train_formatted, label_formatted)
    #except ValueError:
    #    print train_formatted
    #    sys.exit(0)
    svm_clf.fit(train_formatted, label_formatted)

    for item in X_test:
        test_formatted.append(X_test[item])

    #temp_knn_results = []
    temp_svm_results = []
    #temp_knn_results = knn_clf.predict(test_formatted)
    temp_svm_results = svm_clf.predict(test_formatted)
     
    for i in range (0, len(X_test.keys())):
        svm_results.append((X_test.keys()[i], temp_svm_results[i]))
        #knn_results.append((X_test.keys()[i], temp_knn_results[i]))
        
     
    return svm_results
    #return knn_results

# run part B
def run(train, test, language, answer):
    global global_language
    global stop_words
    global punctuation_unicode
    global unigram_tagger
    global bigram_tagger
    global trigram_tagger
    global stemmer
    global window_size


    global_language = language

    if global_language == "english":
        window_size = 3
    elif global_language == "spanish":
        window_size = 2
    else:
        window_size = 3

    global_language = language.lower()
    
    if global_language == "english" or global_language == "spanish":
        stop_words = [word.lower() for word in stopwords.words(global_language)]
    else:
       stop_words =  [u'\ufeffa', u'abans', u'algun', u'alguna', u'algunes', u'alguns', u'altre', u'amb', u'ambd\xf3s', u'anar', u'ans', u'aquell', u'aquelles', u'aquells', u'aqu\xed', u'bastant', u'b\xe9', u'cada', u'com', u'consegueixo', u'conseguim', u'conseguir', u'consigueix', u'consigueixen', u'consigueixes', u'dalt', u'de', u'des de', u'dins', u'el', u'elles', u'ells', u'els', u'en', u'ens', u'entre', u'era', u'erem', u'eren', u'eres', u'es', u'\xe9s', u'\xe9ssent', u'est\xe0', u'estan', u'estat', u'estava', u'estem', u'esteu', u'estic', u'ets', u'fa', u'faig', u'fan', u'fas', u'fem', u'fer', u'feu', u'fi', u'haver', u'i', u'incl\xf2s', u'jo', u'la', u'les', u'llarg', u'llavors', u'mentre', u'meu', u'mode', u'molt', u'molts', u'nosaltres', u'o', u'on', u'per', u'per', u'per que', u'per\xf2', u'perqu\xe8', u'podem', u'poden', u'poder', u'podeu', u'potser', u'primer', u'puc', u'quan', u'quant', u'qui', u'sabem', u'saben', u'saber', u'sabeu', u'sap', u'saps', u'sense', u'ser', u'seu', u'seus', u'si', u'soc', u'solament', u'sols', u'som', u'sota', u'tamb\xe9', u'te', u'tene', u'tenim', u'tenir', u'teniu', u'teu', u'tinc', u'tot', u'\xfaltim', u'un', u'un', u'una', u'unes', u'uns', u'\xfas', u'va', u'vaig', u'van', u'vosaltres', u'']


    if global_language == "catalan":
        stemmer = SnowballStemmer("spanish")
    else:
        stemmer = SnowballStemmer(global_language)    
    
    #train the taggers here and keep the reference global

    if global_language == "english":

        _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        trigram_tagger = load(_POS_TAGGER)


    elif global_language == "spanish":
        train_sents = nltk.corpus.cess_esp.tagged_sents()
        unigram_tagger = nltk.UnigramTagger(train_sents)
        bigram_tagger = nltk.BigramTagger(train_sents, backoff = unigram_tagger)
        trigram_tagger = nltk.TrigramTagger(train_sents, backoff = bigram_tagger)

    else:
        train_sents = nltk.corpus.cess_cat.tagged_sents()
        unigram_tagger = nltk.UnigramTagger(train_sents)
        bigram_tagger = nltk.BigramTagger(train_sents, backoff = unigram_tagger)
        trigram_tagger = nltk.TrigramTagger(train_sents, backoff = bigram_tagger)

    #we have the taggers ready now. use them to extract the features!

    punctuation = list(string.punctuation)   
    punctuation_unicode = [unicode(punc) for punc in punctuation]
    punctuation_unicode.append(unicode("''")) 
    results = {}

    for lexelt in train:
        #print lexelt
        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)
