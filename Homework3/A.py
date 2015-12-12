from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import sys
# don't change the window size
window_size = 10
import nltk
# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}
    temp_list = []
    set_1 = set()
    for item in data:
        set_1 = set()
        temp_list = []
        for val in data[item]:
            #print val[0]
            #print val[1]
            
            left_context_tokens = nltk.word_tokenize(val[1])
            set_1.update(set(left_context_tokens[-10:]))
            '''for word in left_context_tokens:
                if word in temp_list:
                    continue
                temp_list.append(word)
                #print word'''
            
            right_context_tokens = nltk.word_tokenize(val[3])
            set_1 = set_1.union(set(right_context_tokens[:10]))
            '''for word in right_context_tokens:
                if word in temp_list:
                    continue
                temp_list.append(word)
            #    print word'''
        
        s[item] = list(set_1)    


    # implement your code here
    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    temp_dict = {}
    word_count = 0
    
    for item in data:
        labels[item[0]] = item[4]

    for item in data:
        temp_list = []
        left_context_tokens = nltk.word_tokenize(item[1])
        right_context_tokens = nltk.word_tokenize(item[3])
        left_context_tokens = left_context_tokens[-10:]
        right_context_tokens = right_context_tokens[:10]

        for word in s:
            word_count = 0
            word_count = word_count + left_context_tokens.count(word)
            word_count = word_count + right_context_tokens.count(word)
            temp_list.append(word_count)

        vectors[item[0]] = temp_list
        
    # implement your code here
    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()
    
    train_formatted = []
    label_formatted = []
    test_formatted = []
    for item in X_train:
        train_formatted.append(X_train[item])
        label_formatted.append(y_train[item])
        #print len(X_train[item])
        #print X_train[item]
        #knn_clf.fit(train_formatted, test_formatted) 
        #print len(X_train[item])
    
    for item in X_test:
        test_formatted.append(X_test[item])
    
    knn_clf.fit(train_formatted, label_formatted)
    svm_clf.fit(train_formatted, label_formatted)

    temp_knn_results = []
    temp_svm_results = []

    temp_knn_results = knn_clf.predict(test_formatted)
    temp_svm_results = svm_clf.predict(test_formatted)

    for i in range (0, len(X_test.keys())):
        svm_results.append((X_test.keys()[i], temp_svm_results[i]))
        knn_results.append((X_test.keys()[i], temp_knn_results[i]))


    # implement your code here

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output
    
    '''
    f = open(output_file, 'w')
    for item in results:
        results[item].sort(key = lambda x: x[1].lower()) 
        for element in results[item]:
            #f.write(str(item) + ' ' + str(element[0]) + ' ' + str(element[1]) + '\n')
            f.write(str(replace_accented(item)) + ' ' + str(replace_accented(element[0])) + ' ' + str(element[1]) + '\n')
            

    f.close()

    f = open(output_file, 'r')
    lines = f.readlines()
    lines.sort()
    
    f.close()
    
    f = open(output_file, 'w')
    for item in lines:
        f.write(item)
    
    f.close()     


    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



