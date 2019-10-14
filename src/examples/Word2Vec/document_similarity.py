import gensim
from gensim import utils
from gensim.models import Doc2Vec
import smart_open
import sys
import argparse
import codecs

# input corpus is one line per documents
# following function will use document name as a label
def read_corpus(fname):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        docId = ""
        docText = ""
        tmp = ""
        for i, line in enumerate(f):
                # For training data, add tags
                tmp = gensim.utils.to_unicode(line).split() 
                docId = tmp[0]
                docText = tmp[1:]
                yield gensim.models.doc2vec.TaggedDocument(docText, [docId])

def train_model(train):
    train = list(read_corpus(train))
    
    model = Doc2Vec(size=30, window=10, min_count=1, workers=11,alpha=0.025, min_alpha=0.025)
    
    model.build_vocab(train) # build vocabulary
    model.train(train, total_examples=model.corpus_count, epochs=model.iter)

    return model


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train/load a document similarity model')

    parser.add_argument('--train', type=str, help='Training file, one document per line prefixed by document name')
    parser.add_argument('--save_model', type=str, help='Same the model with this name')
    parser.add_argument('--load_model', type=str, help='Load an already trained model')

    params = parser.parse_args()

    if params.load_model != None:
        print ("Loading model ... ...")
        model = Doc2Vec.load(params.load_model)
    elif params.train == None:
        print ("Enter a training set")
        exit()

    if params.load_model == None:
        model = train_model(params.train) # train the model

    if params.save_model != None:
        model.save(params.save_model)
        print ("Model Saved")

    while True:
        word = input("Enter a word to find its nearest neighbours")
        print (model.most_similar(word))
"""


