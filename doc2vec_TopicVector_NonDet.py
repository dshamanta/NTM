#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via the distributed memory and distributed bag of words models from
[1]_, using either hierarchical softmax or negative sampling [2]_ [3]_.

**Install Cython with `pip install cython` before installing gensim, to use optimized
doc2vec training** (70x speedup [blog]_).

Initialize a model with e.g.::

>>> model = Doc2Vec(sentences, size=100, window=8, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Doc2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Doc2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

.. [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
.. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [blog] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/

"""

import logging
import os
import time
import pickle


from numpy import array, dot, reshape, matrix, zeros, argsort, random, empty, exp, divide, uint32, dtype, argmax, multiply, float32 as REAL, sum as np_sum
from scipy.special import expit
try:
    from queue import PriorityQueue
except ImportError:
    from Queue import PriorityQueue

logger = logging.getLogger("doc2vec_Det_TopicVector")
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from word2vec import Word2Vec, Vocab, train_cbow_pair, train_sg_pair

FAST_VERSION = -1

def train_sentence_dbow(model, sentence, lbls, alpha, work=None, train_words=True, train_lbls=True):
	"""
	Update distributed bag of words model by training on a single sentence.

	The sentence is a list of Vocab objects (or None, where the corresponding
	word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

	This is the non-optimized, Python version. If you have cython installed, gensim
	will use the optimized version from doc2vec_inner instead.

	"""
	neg_labels = []
	if model.negative:
		# precompute negative labels
		neg_labels = zeros(model.negative + 1)
		neg_labels[0] = 1.0

	for label in lbls:
		if label is None:
			continue  # OOV word in the input sentence => skip
		for word in sentence:
			if word is None:
				continue  # OOV word in the input sentence => skip
			train_sg_pair(model, word, label, alpha, neg_labels, train_words, train_lbls)

	return len([word for word in sentence if word is not None])

def train_sentence_dm(model, sentence, lbls, alpha, work=None, neu1=None, train_words=True, train_lbls=True):
    """
    Update distributed memory model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from doc2vec_inner instead.

    """
    
    lbl_indices = [lbl.index for lbl in lbls if lbl is not None]

    if(len(lbl_indices) <= model.K):return 0

    docIndxPos = int(model.index2word[lbl_indices[0]][5:])
    topKTopics = argsort(model.w_ld[docIndxPos])[::-1][:4]

    
    selected_lbl_indices = [lbl_indices[0]];
    for i in range(2):
        selected_lbl_indices.append(lbl_indices[topKTopics[i]+1])

    
    lbl_sum = np_sum(model.syn0[lbl_indices[0]], axis=0)
##    lbl_len = len(lbl_indices)
    lbl_len = 1
    neg_labels = []
    if model.negative:
        # precompute negative labels
        neg_labels = zeros(model.negative + 1)
        neg_labels[0] = 1.

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        reduced_window = random.randint(model.window)  # `b` in the original doc2vec code
        start = max(0, pos - model.window + reduced_window)
        window_pos = enumerate(sentence[start : pos + model.window + 1 - reduced_window], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
        
        l1 = np_sum(model.syn0[word2_indices], axis=0) + lbl_sum  # 1 x layer1_size
        if word2_indices and model.cbow_mean:
            l1 /= (len(word2_indices) + lbl_len)
        neu1e = train_cbow_pair(model, word, word2_indices, l1, alpha, neg_labels, train_words, train_words)
        if train_lbls:
            model.syn0[selected_lbl_indices[0]] += neu1e
            model.syn0[selected_lbl_indices[1:]] += (neu1e/model.noOfLabels)
            
        word2_indices.append(word.index)
        a_1 = np_sum(model.syn0[word2_indices], axis=0)/len(word2_indices)

        
        docIndxNeg = selectNegativeDocs(docIndxPos)
        
        myTrain(model, docIndxPos, docIndxNeg, a_1)

    return len([word for word in sentence if word is not None])


def selectNegativeDocs(docIndxPos):
##    TODO: design data structure to select the docs that does not contains these wordIndx in the order
    docIndxNeg = 0;
##    if (docIndxPos < 100):
##        docIndxNeg = random.randint(100, model.noOfLabels)
##    elif (docIndxPos >= 100 and docIndxPos < 200):
##        t = [random.randint(0, 100), random.randint(200, model.noOfLabels)]
##        docIndxNeg = t[random.randint(0, 2)];
##    else:
##        docIndxNeg = random.randint(0, 200)
#    posTopic = argmax(model.w_ld[docIndxPos]);
#    docIndxNeg = random.randint(0, model.noOfLabels)
#    while(posTopic == argmax(model.w_ld[docIndxNeg])):
#        docIndxNeg = random.randint(0, model.noOfLabels)
#    print model.noOfLabels
#    raw_input('Press any key to continue')
    docIndxNeg = random.randint(0, model.noOfLabels)
#    print docIndxNeg
#    print model.target[docIndxNeg]
#    raw_input('Press any key to continue')
    while(model.target[docIndxPos] == model.target[docIndxNeg]):
        docIndxNeg = random.randint(0, model.noOfLabels)
#        print docIndxNeg
#        print model.target[docIndxNeg]
#        raw_input('Press any key to continue')
    return docIndxNeg

def myTrain(model, docIndxPos, docIndxNeg, ngramEmbeddings, alpha = 0.1):

    a_2 = expit(reshape(ngramEmbeddings, (1, ngramEmbeddings.size))*matrix(model.w_lt))

##    sum_a_2 = exp(a_2).sum()
##    a_2 = divide(exp(a_2), sum_a_2)

    a_3Pos = expit(matrix(model.w_ld[docIndxPos])*(a_2.transpose()))
    a_3Neg = expit(matrix(model.w_ld[docIndxNeg])*(a_2.transpose()))
    
##    a_3Pos = matrix(model.w_ld[docIndxPos])*(a_2.transpose())
##    a_3Neg = matrix(model.w_ld[docIndxNeg])*(a_2.transpose())


   ## cost = max(0, 0.5 - a_3Pos + a_3Neg);
##    if(cost > 0):
    err3Pos = (1 - a_3Pos)
    err3Neg = (0 - a_3Neg)

    err2Pos = multiply((err3Pos*model.w_ld[docIndxPos]), multiply(a_2, (1-a_2)))
    err2Neg = multiply((err3Neg*model.w_ld[docIndxNeg]), multiply(a_2, (1-a_2)))
    
    model.w_ld[docIndxPos] = model.w_ld[docIndxPos] + alpha*err3Pos*a_2
    model.w_ld[docIndxNeg] = model.w_ld[docIndxNeg] + alpha*err3Neg*a_2
        
    model.w_lt = model.w_lt + alpha*(matrix(ngramEmbeddings).transpose())*err2Pos
    model.w_lt = model.w_lt + alpha*(matrix(ngramEmbeddings).transpose())*err2Neg
    return

class LabeledSentence(object):
    """
    A single labeled sentence = text item.
    Replaces "sentence as a list of words" from Word2Vec.

    """
    def __init__(self, words, labels):
        """
        `words` is a list of tokens (unicode strings), `labels` a
        list of text labels associated with this text.

        """
        self.words = words
        self.labels = labels

    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.words, self.labels)


class Doc2Vec(Word2Vec):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""
    def __init__(self, sentences=None, size=300, alpha=0.025, window=8, min_count=5,
                 sample=0, seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1, negative=0,
                 dm_mean=0, train_words=True, train_lbls=True, **kwargs):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        LabeledSentence object that will be used for training.

        The `sentences` iterable can be simply a list of LabeledSentence elements, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `dm` defines the training algorithm. By default (`dm=1`), distributed memory is used.
        Otherwise, `dbow` is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator.

        `min_count` = ignore all words with total frequency lower than this.

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 0 (off), useful value is 1e-5.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0).

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).

        `dm_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
        Only applies when dm is used.

        """
        Word2Vec.__init__(self, size=size, alpha=alpha, window=window, min_count=min_count,
                          sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                          sg=(1+dm) % 2, hs=hs, negative=negative, cbow_mean=dm_mean, **kwargs)
        self.train_words = train_words
        self.train_lbls = train_lbls
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    @staticmethod
    def _vocab_from(self, sentences):
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at item #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
            sentence_length = len(sentence.words)
            for label in sentence.labels:
                total_words += 1
                if label in vocab:
                    vocab[label].count += sentence_length
                else:
                    vocab[label] = Vocab(count=sentence_length)
            for word in sentence.words:
                total_words += 1
                if word in vocab:
                    vocab[word].count += 1
                else:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i word types from a corpus of %i words and %i items" %
                    (len(vocab), total_words, sentence_no + 1))
        self.noOfLabels = sentence_no + 1
        return vocab

    def _prepare_sentences(self, sentences):
        for sentence in sentences:
            # avoid calling random_sample() where prob >= 1, to speed things up a little:
            sampled = [self.vocab[word] for word in sentence.words
                       if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or
                                                  self.vocab[word].sample_probability >= random.random_sample())]
            yield (sampled, [self.vocab[word] for word in sentence.labels if word in self.vocab])

    def _get_job_words(self, alpha, work, job, neu1):
        if self.sg:
            return sum(train_sentence_dbow(self, sentence, lbls, alpha, work, self.train_words, self.train_lbls) for sentence, lbls in job)
        else:
            return sum(train_sentence_dm(self, sentence, lbls, alpha, work, neu1, self.train_words, self.train_lbls) for sentence, lbls in job)

    def __str__(self):
        return "Doc2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2word), self.layer1_size, self.alpha)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])  # don't bother storing the cached normalized vectors
        super(Doc2Vec, self).save(*args, **kwargs)

    def reset_weights_topic_model(self):
        """Reset all projection weights to an initial (untrained) state"""
        logger.info("resetting topic-ngram and document-topic layers weights...")

        self.w_lt = empty((self.layer1_size, self.K), dtype=REAL)
        self.w_ld = empty((self.noOfLabels, self.K), dtype=REAL)
        self.word_impact = empty((len(self.index2word), self.K, self.layer1_size), dtype=REAL)

        random.seed(uint32(self.hashfxn(str(self.seed))))
        self.w_lt = random.rand(self.layer1_size, self.K)
        random.seed(uint32(self.hashfxn(str(self.seed))))
        self.w_ld = random.rand(self.noOfLabels, self.K) - 0.5
#        for i in range(self.noOfLabels):
#            random.seed(uint32(self.hashfxn(str(i) + str(self.seed))))
#            self.w_ld[i] = random.rand(1, self.K) - 0.5
        
        logger.info( 'Word-Topic Weights: RowXColumn: %i %i'%(len(self.w_lt), len(self.w_lt[0])) )
        logger.info( 'Topic-Document Weights: RowXColumn: %i %i'%(len(self.w_ld), len(self.w_ld[0])) )

    


class LabeledLineSentence(object):
    """Simple format: one sentence = one line = one LabeledSentence object.

    Words are expected to be already preprocessed and separated by whitespace,
    labels are constructed automatically from the sentence line number."""
    def __init__(self, source):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.topics_name = []
        for i in range(20):
            self.topics_name.append('TOPIC_%s'%i)

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield LabeledSentence(utils.to_unicode(line).split(), ['SENT_%s' % item_no] + self.topics_name )
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), ['SENT_%s' % item_no] + self.topics_name )
                    
def jaccard_similarity(target, predict, nDoc = 100):
    a_and_b = 0.0;
    a_or_b = 0.0;

    for i in range(nDoc):
        for j in range(i+1, nDoc):
            a_val = 0
            b_val = 0
            a_or_b = a_or_b + 1
            if(predict[i] == predict[j]):
                a_val = 1
            if(target[i] == target[j]):
                b_val = 1
            if(a_val == 1 and b_val == 1):
                a_and_b = a_and_b + 1
            if(a_val == 0 and b_val == 0):
                a_or_b = a_or_b - 1

    similarity = a_and_b / a_or_b
    return similarity

def wordInfluenceOnTopics(model, noOfWords = 25):
    with open ('../Data/topic_words.txt', 'w') as fout:
        for t in range(model.K):
            fout.write ('================ TOPIC: %s ==============\n'% t)
            pq = PriorityQueue()
            for v in range(len(model.vocab)):
                word = model.index2word[v]
                if(('SENT' not in word) and ('TOPIC' not in word)):
                    vec_word = model.word_impact[v][t]
                    similarity = dot(matutils.unitvec(vec_word), matutils.unitvec(model['TOPIC_'+str(t)]))
                    pq.put((similarity, word))
            for i in range(noOfWords):
#                print pq.get()
                fout.write(str(pq.get()))
                fout.write('\n')
    
def measuringClassificationAccuracy(model, epoch):
    pred = empty(model.noOfLabels)
    pred = argmax(model.w_ld, axis = 1)
    
    filename = '../Data/TopicVectors_Iter/Pred_'+str(epoch)+'.txt';
    with open(filename, 'w') as fout:
        for i in range(model.noOfLabels):
            fout.write(str(pred[i]) + '\n')
        

    similarity = jaccard_similarity(target = model.target, predict = pred, nDoc = model.noOfLabels)
    return similarity

def normalizeTheChanges(model):
    for t in range(model.K):
        topic = 'TOPIC_'+str(i)
        delVec = abs(model[topic] - model.init_w_topic[t])
        for v in range(len(model.vocab)):
            word = model.index2word[v]
            if(('SENT' not in word) and ('TOPIC' not in word)):
                model.word_impact[v][t] /= delVec   
def saveTopicVectorsInEachIteration(model, epoch):
    filename = '../Data/TopicVectors_Iter/full20news_topic_vectors_iter'+str(epoch)+'.txt'
    with open(filename, 'w+') as fout:
        for i in range(model.K):
            t = 'TOPIC_'+str(i)
            fout.write(t + ' ')
            for j in range(100):
                v = model[t][j]
                fout.write(str(v)+' ')
            fout.write('\n')

def getNoOfDoc(filename):
    with open(filename) as f:
        noOfDoc =  sum(1 for _ in f)
    return noOfDoc
# Example: ./word2vec.py ~/workspace/word2vec/text8 ~/workspace/word2vec/questions-words.txt ./text8
if __name__ == "__main__":
    start = time.clock()
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("using optimization %s" % FAST_VERSION)
##        sentences = LabeledLineSentence('./debug_data.txt')
    
    model = Doc2Vec(size=100, alpha=0.025, min_alpha = 0.025, iter = 5, min_count = 1, ntopic = 20)
    
    noOfDoc = getNoOfDoc('../Data/Full20News_20groups_target.txt')
    model.target = zeros((noOfDoc,), dtype=int)
    with open("../Data/Full20News_20groups_target.txt") as f:
        for i, x in enumerate(f):
            model.target[i] = int(x)
    
#    model.target = zeros((400,), dtype=int)
#    for i in range(400):
#        model.target[i] = i / 100
    
    sentences = LabeledLineSentence('../Data/Full20News_20groups_data.txt')
#    sentences = LabeledLineSentence('../Data/synthetic_data_5.txt')

    model.build_vocab(sentences)
    
#    initial value of topic vectors. It is saved because of watching the change after completing training
    model.init_w_topic = empty((model.K, model.layer1_size), dtype=REAL)
    for i in range(model.K):
        t = 'TOPIC_'+str(i)
        model.init_w_topic[i] = model[t]
    
#   Classification accuracy after random assignment
    similarity = measuringClassificationAccuracy(model, 0)
    print ("Epoch 0:\tSimilarity: %f" %(similarity))
    
    for epoch in range(20):
        logging.info("=====================Starting a new epoch=====================")
        saveTopicVectorsInEachIteration(model, epoch)        
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        if(model.alpha <0.002):
            model.alpha = 0.002
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        similarity = measuringClassificationAccuracy(model, epoch)
        print ("Epoc: %d\tSimilarity: %f" %(epoch+1, similarity))
        
    print("Time required for completing the task: %f" % (time.clock() - start))

    normalizeTheChanges(model)
    wordInfluenceOnTopics(model)
#    model.save_word2vec_format('../Full20News_vectors.txt')
