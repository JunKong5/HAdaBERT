# ======================================= #
# --------------- DataModel ------------- #
# ======================================= #
import pprint
import os
import csv
import sys
import re
import torch

from datasets.utils import InputExample

pp = pprint.PrettyPrinter(indent=4)
from datasets.utils import SentenceProcessor

class YELP_13(SentenceProcessor):
    NAME = 'YELP_13'
    NUM_CLASSES = 5
    IS_MULTILABEL = False

    def __init__(self, data_dir='corpus'):
        super().__init__()
        self.d_train = self._read_tsv(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.train.ss'))
        self.d_dev = self._read_tsv(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.dev.ss'))
        self.d_test = self._read_tsv(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.test.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sentences(self):
        train = self._create_sentences(self.d_train)
        dev = self._create_sentences(self.d_dev)
        test = self._create_sentences(self.d_test)
        return tuple([train, dev, test])

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)

            text = clean_document(line[6])
            label = int(line[4])-1
            #if i == 50:
            #   break
            print("\r{:>6}".format(i), end="")
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _create_sentences(self, *datasets):
        sentences = []
        for dataset in datasets:
            print()
            for id, document in enumerate(dataset):
                review = document[6]
                label = int(document[4])-1
                sentences.extend([InputExample( text=sentence, label=label) for
                                  sentence in generate_sents(clean_document(review))])
                print("\r{:>6}".format(id), end="")
                if id==50:
                    break


        return sentences



class SST2(object):
    NAME = 'SST2'
    NUM_CLASSES = 2
    IS_MULTILABEL = False
    def __init__(self, data_dir='corpus'):
        super().__init__()
        self.d_train = self._read_file(os.path.join(data_dir, 'sst2', 'sentiment-train'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'sst2', 'sentiment-dev'))
        self.d_test = self._read_file(os.path.join(data_dir, 'sst2', 'sentiment-test'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def _read_file(self, dataset):
        with open(dataset, "r",encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    def _create_examples(self, documents, type):

        def clean_document(document):
            string = re.sub(r"<sssss>", "", document)
            string = re.sub(r" n't", "n't", string)
            string = re.sub(r"[^A-Za-z0-9(),!?\'.`]", " ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.lower().strip()

        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            # text = [sentence for sentence in split_sents(line[2])]
            #print(line[0])
            text = clean_document(line[0])
            # print(line[1])
            label = int(line[1])
            #print(label)
            #if i == 10:
                # print(text)
            #    break
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples






class IMDB(SentenceProcessor):
    NAME = 'IMDB'
    NUM_CLASSES = 10
    IS_MULTILABEL = False

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_tsv(os.path.join(data_dir, 'IMDB1', 'train.tsv'))
        self.d_dev = self._read_tsv(os.path.join(data_dir, 'IMDB1', 'dev.tsv'))
        self.d_test = self._read_tsv(os.path.join(data_dir, 'IMDB1', 'test.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sentences(self):
        train = self._create_sentences(self.d_train)
        dev = self._create_sentences(self.d_dev)
        test = self._create_sentences(self.d_test)
        return tuple([train, dev, test])

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)

            text = clean_document(line[1])

            label = list(line[0])
            label = list(map(int, label))
            label = torch.argmax(torch.tensor(label, dtype=torch.int)).item()
            # if i == 50:
            #     break
            print("\r{:>6}".format(i), end="")
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _create_sentences(self, *datasets):
        sentences = []
        for dataset in datasets:
            for id, document in enumerate(dataset):
                review = document[1]
                label = list(document[0])
                label = list(map(int, label))
                label = torch.tensor(label)
                label = torch.argmax(torch.tensor(label, dtype=torch.int))
                sentences.extend([InputExample( text=sentence, label=label) for
                                  sentence in generate_sents(clean_document(review))])
                print("\r{:>6}".format(id), end="")
                # if id==50:
                #    break

        return sentences


class aapd(SentenceProcessor):
    NAME = 'aapd'
    NUM_CLASSES = 54
    IS_MULTILABEL = True

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_tsv(os.path.join(data_dir, 'aapd', 'train.tsv'))
        self.d_dev = self._read_tsv(os.path.join(data_dir, 'aapd', 'dev.tsv'))
        self.d_test = self._read_tsv(os.path.join(data_dir, 'aapd', 'test.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sentences(self):
        train = self._create_sentences(self.d_train)
        dev = self._create_sentences(self.d_dev)
        test = self._create_sentences(self.d_test)
        return tuple([train, dev, test])

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)

            text = clean_document(line[1])
            label = line[0]

            #if i ==50:
            #break
            print("\r{:>6}".format(i), end="")

            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _create_sentences(self, *datasets):
        sentences = []
        for dataset in datasets:
            print()
            for id, document in enumerate(dataset):
                review = document[1]

                label = list(document[0])
                label = list(map(int, label))
                label = torch.tensor(label)

                sentences.extend([InputExample( text=sentence, label=label) for
                                  sentence in generate_sents(clean_document(review))])
                print("\r{:>6}".format(id), end="")
                #if id==50:
                #    break


        return sentences


class reuters(SentenceProcessor):
    NAME = 'reuters'
    NUM_CLASSES = 90
    IS_MULTILABEL = True

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_tsv(os.path.join(data_dir, 'reuters', 'train.tsv'))
        self.d_dev = self._read_tsv(os.path.join(data_dir, 'reuters', 'dev.tsv'))
        self.d_test = self._read_tsv(os.path.join(data_dir, 'reuters', 'test.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sentences(self):
        train = self._create_sentences(self.d_train)
        dev = self._create_sentences(self.d_dev)
        test = self._create_sentences(self.d_test)
        return tuple([train, dev, test])

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)

            text = clean_document(line[1])
            label = line[0]

            if i ==50:
             break
            print("\r{:>6}".format(i), end="")

            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _create_sentences(self, *datasets):
        sentences = []
        for dataset in datasets:
            print()
            for id, document in enumerate(dataset):
                review = document[1]

                label = list(document[0])
                label = list(map(int, label))
                label = torch.tensor(label)

                sentences.extend([InputExample( text=sentence, label=label) for
                                  sentence in generate_sents(clean_document(review))])
                print("\r{:>6}".format(id), end="")
                if id==50:
                    break


        return sentences


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"sssss", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


def clean_document(document):
    string = re.sub(r"<sssss>", "", document)
    string = re.sub(r" n't", "n't", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'.`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()



def generate_sents(docuemnt, max_length=160):
    if isinstance(docuemnt, list):
        docuemnt = docuemnt[0]
    string = re.sub(r"[!?]", " ", docuemnt)
    string = re.sub(r"\.{2,}", " ", string)
    sents = string.strip().split('.')
    sents = [clean_string(sent) for sent in sents]
    n_sents = []
    n_sent = []
    for sent in sents:
        n_sent.extend(sent)
        if len(n_sent) > max_length:
            n_sents.append(" ".join(n_sent))
            n_sent = []
            n_sent.extend(sent)
    n_sents.append(" ".join(n_sent))
    return n_sents
