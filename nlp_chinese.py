# -*- coding: utf-8 -*-
import logging
import multiprocessing
import os
import re

import gensim.corpora
import gensim.models
import gensim.models.word2vec
import gensim.models.doc2vec
import jieba
import opencc  # pip install OpenCC
import six

logger = logging.getLogger()


def process_corpus_extraction(file_input, file_output):
    with open(file_output, 'w') as f_out:
        wiki = gensim.corpora.WikiCorpus(
            file_input, lemmatize=False, dictionary={})
        num_total = 0
        for num, text in enumerate(wiki.get_texts()):
            line = b' '.join(text).decode('utf-8')\
                if six.PY3 else ' '.join(text)
            f_out.writelines([line])
            num_total = num + 1
            if num_total % 10000 == 0:
                logger.info("Saved " + str(num_total) + " articles")
        logger.info("Finished, Saved " + str(num_total) + " articles")


def process_chinese_filtering(file_input, file_output):
    with open(file_input, 'r') as f_in, open(file_output, 'w') as f_out:
        rule = re.compile(r'[ a-zA-z]')  # delete english char and blank
        num_total = 0
        for num, line in enumerate(f_in):
            f_out.writelines([rule.sub('', line)])
            num_total = num + 1
            if num_total % 10000 == 0:
                logger.info("Saved " + str(num_total) + " lines")
        logger.info("Finished, Saved " + str(num_total) + " lines")


def process_chinese_transformation(file_input, file_output, mode='t2s'):
    with open(file_input, 'r') as f_in, open(file_output, 'w') as f_out:
        config_mode = mode + '.json'
        num_total = 0
        for num, line in enumerate(f_in):
            f_out.writelines([
                opencc.convert(line, config=config_mode)])
            num_total = num + 1
            if num_total % 10000 == 0:
                logger.info('Converted %s lines' % num_total)
        logger.info('Finished, Converted %s lines' % num_total)


def process_chinese_segmentation(file_input, file_output, split=' '):
    with open(file_input, 'r') as f_in, open(file_output, 'w') as f_out:
        num_total = 0
        for num, line in enumerate(f_in):
            f_out.writelines([split.join(jieba.cut(line))])
            num_total = num + 1
            if num_total % 10000 == 0:
                logger.info('Segmented %s lines' % num_total)
        logger.info('Finished, Segmented %s lines' % num_total)


def process_doc_training(file_input, file_output):
    model = gensim.models.Doc2Vec(
        gensim.models.doc2vec.TaggedLineDocument(file_input),
        size=400, workers=multiprocessing.cpu_count())
    model.save(file_output)


def process_word_training(file_input, file_output):
    model = gensim.models.Word2Vec(
        gensim.models.word2vec.LineSentence(file_input),
        size=400, workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
    model.save(file_output)


def train_model(file_input, file_output):
    file_intermediate = os.path.join(
        os.path.dirname(file_input),
        os.path.splitext(file_input)[0])
    process_corpus_extraction(
        file_input, file_intermediate + '.extracted')
    process_chinese_filtering(
        file_intermediate + '.extracted',
        file_intermediate + '.filtered')
    process_chinese_transformation(
        file_intermediate + '.filtered',
        file_intermediate + '.transformed')
    process_chinese_transformation(
        file_intermediate + '.transformed',
        file_intermediate + '.segmented')
    # we can train for either word2vec or doc2vec
    # process_word_training(
    #     file_intermediate + '.segmented', file_output)
    process_doc_training(
        file_intermediate + '.segmented', file_output)


if __name__ == '__main__':
    # train for model
    # train_model(
    #     'zhwiki-latest-pages-articles.xml.bz2',
    #     'corpus.zhwiki.doc.model')

    # test for model
    model = gensim.models.Doc2Vec.load('model/corpus.zhwiki.doc.model')
    similarity_1 = model.docvecs.similarity_unseen_docs(
        model, jieba.cut(u"晚饭很好吃"), jieba.cut(u'你吃了吗'))
    similarity_2 = model.docvecs.similarity_unseen_docs(
        model, jieba.cut(u"滚"), jieba.cut(u'呵呵'))
    print(similarity_1, similarity_2)
