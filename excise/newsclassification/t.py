# -*- coding: utf-8 -*

#
#参考http://segmentfault.com/a/1190000002472791

import jieba
import numpy
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB



def input_data(train_file, test_file):
    i = 1
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    
    with open(train_file, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 3)
            i +=1
            print (str(i))
            train_words.append(tks[3].decode('gb18030'))
            train_tags.append(tks[0].decode('gb18030'))
    with open(test_file, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 3)
            test_words.append(tks[3].decode('gb18030'))
            test_tags.append(tks[0].decode('gb18030'))
    return train_words, train_tags, test_words, test_tags


# with open('stopwords.txt', 'r') as f:
#     stopwords = set([w.strip() for w in f])
comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)


def vectorize(train_words, test_words):
    v = HashingVectorizer(tokenizer=comma_tokenizer, n_features=30000, non_negative=True)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    return train_data, test_data


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred)
    m_recall = metrics.recall_score(actual, pred)
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)


def train_mnb(train_data, train_tags):
    mnb = MultinomialNB(alpha=0.01)
    mnb.fit(train_data, numpy.asarray(train_tags))
    return mnb


def main():
    train_file = 'data/train'
    test_file = 'data/test'
    train_words, train_tags, test_words, test_tags = input_data(train_file, test_file)
    train_data, test_data = vectorize(train_words, test_words)
    mnb = train_mnb(train_data, train_tags)
    pred = mnb.predict(test_data)
    evaluate(numpy.asarray(test_tags), pred)


if __name__ == '__main__':
    main()