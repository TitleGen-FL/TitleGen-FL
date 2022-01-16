import torch
from torchtext import data
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import jieba
import nltk
import word2vec
import re
def x_tokenize(x):
    str = x
    return nltk.word_tokenize(str)

TEXT = data.Field(sequential=True, tokenize=x_tokenize,fix_length=100,
            use_vocab=True)
LABEL = data.Field(sequential=False,
            use_vocab=False)

train, dev, test = data.TabularDataset.splits(path='dataset_train_DL_model',
                                              train='csv_train.csv',
                                              validation='csv_dev.csv',
                                              test='csv_test.csv',
                                              format='csv',
                                              skip_header=True,
                                              csv_reader_params={'delimiter':','},
                                              fields=[('text',TEXT),('label',LABEL)])

TEXT.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits((train,dev,test),
                                                             batch_size = 256,
                                                             shuffle=True,
                                                             sort=False,
                                                             sort_within_batch=False,
                                                             repeat=False)

def getTEXT():
    return TEXT
def getLabel():
    return LABEL
def getIter():
    return train_iter, val_iter, test_iter