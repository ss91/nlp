import random
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
from providedcode.dependencygraph import DependencyGraph
import sys


#assumes that english.model exists and doesn't error check

if __name__ == "__main__":

    tp = TransitionParser.load('english.model')
    sentences = []
    for line in sys.stdin:
        line = line.strip()
        sentence = DependencyGraph.from_sentence(line)
        sentences.append(sentence)
        #print sentence
    
    parsed = tp.parse(sentences)
    for p in parsed:
        print p.to_conll(10).encode('utf-8')
        #prints to stdout which is redirected anyway...    
        #with open ('new.conll', 'w') as f:
        #    f.write(parsed[0].to_conll(10).encode('utf-8'))
        #    f.write('\n')

