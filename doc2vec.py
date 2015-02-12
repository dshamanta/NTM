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

from Queue import Queue

from numpy import array, reshape, matrix, zeros, random, empty, uint32, dtype, float32 as REAL, sum as np_sum
from scipy.special import expit

logger = logging.getLogger("doc2vec")
from gensim import utils  # utility fnc for pickling, common scipy operations etc
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

    if(len(lbl_indices) <= 0):return 0
        
    lbl_sum = np_sum(model.syn0[lbl_indices], axis=0)
    lbl_len = len(lbl_indices)
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
            model.syn0[lbl_indices] += neu1e
            
        word2_indices.append(word.index)
        a_1 = np_sum(model.syn0[word2_indices], axis=0)/len(word2_indices)

        docIndxPos = int(model.index2word[lbl_indices[0]][5:])
        myTrain(model, docIndxPos, a_1, alpha, 1)
        docIndxNeg = selectNegativeDocs(docIndxPos, word2_indices)
        myTrain(model, docIndxNeg, a_1, alpha, 0)
             

    return len([word for word in sentence if word is not None])

def selectNegativeDocs(docIndxPos, wordIndx):
##    TODO: design data structure to select the docs that does not contains these wordIndx in the order
    docIndxNeg = 0;
    if (docIndxPos < 584):
        docIndxNeg = random.randint(585, model.noOfLabels)
    elif (docIndxPos >= 584 and docIndxPos < 1184):
        t = [random.randint(0, 584), random.randint(1184, model.noOfLabels)]
        docIndxNeg = t[random.randint(0, 2)];
    else:
        docIndxNeg = random.randint(0, 1184)
    return docIndxNeg

def myTrain(model, docIndx, ngramEmbeddings, alpha = 0.01, positive = 1):
    a_2 = expit(reshape(ngramEmbeddings, (1, ngramEmbeddings.size))*matrix(model.w_lt))
    a_3 = (a_2)*(matrix(model.w_ld[docIndx]).transpose())

    if(positive == 1):
        err3 = (1 - a_3)
    else:
        err3 = (0 - a_3)

    tst = err3*model.w_ld[docIndx]
    err2 = array(tst)* array(a_2)

    model.w_ld[docIndx] = model.w_ld[docIndx] + alpha*err3*a_2
    if(min(model.w_ld[docIndx, :]) < 0):
        model.w_ld[docIndx, :] -= min(model.w_ld[docIndx, :])
    model.w_ld[docIndx, :] = model.w_ld[docIndx, :] / sum(model.w_ld[docIndx, :])
            
    model.w_lt = model.w_lt + alpha*(matrix(ngramEmbeddings).transpose())*err2
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
##        logger.info(self.layer1_size)
        self.w_lt = empty((self.layer1_size, self.K), dtype=REAL)
        self.w_ld = empty((self.noOfLabels, self.K), dtype=REAL)
        epsilon_int = 0.12
##        random.seed(uint32(self.hashfxn(str(self.seed))))
        self.w_lt = (random.rand(self.layer1_size, self.K)*2*epsilon_int - epsilon_int)
##    w_ld = (random.rand(n_docs, n_topic)*2*epsilon_int - epsilon_int)
##    w_lt = random.rand(layer1_size, n_topic)
        self.w_ld = random.rand(self.noOfLabels, self.K)

        for i in range(self.noOfLabels):
            self.w_ld[i, :] = self.w_ld[i, :]/sum(self.w_ld[i, :])

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

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield LabeledSentence(utils.to_unicode(line).split(), ['SENT_%s' % item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), ['SENT_%s' % item_no])

# Example: ./word2vec.py ~/workspace/word2vec/text8 ~/workspace/word2vec/questions-words.txt ./text8
if __name__ == "__main__":
        start = time.clock()
        logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
        logging.info("using optimization %s" % FAST_VERSION)
        
##        sentences = LabeledLineSentence('./data/abstracts_1984.txt')
        sentences = LabeledLineSentence('./ntm_3group_news.txt')
        model = Doc2Vec(size=100, alpha=0.025, min_alpha = 0.025, iter = 5)
        model.build_vocab(sentences)
##        model.load_word2vec_format('vectors.txt')

        for epoch in range(5):
                logging.info("=====================Starting a new epoch=====================")
                model.train(sentences)
                model.alpha -= 0.002  # decrease the learning rate
                model.min_alpha = model.alpha  # fix the learning rate, no decay
        print("Time required for completing the task: %f" % (time.clock() - start))
        
