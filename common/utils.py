from datasets.utils import Data
from cfgs.constants import DATASET_MAP, DATASET_PATH_MAP
from torch.utils.data import DataLoader
from cfgs.constants import SAVED_MODEL_PATH, LABLES
import torch
import os
from models.bert_adaptive_encoder import *

import re

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
import math

inf = math.inf
nan = math.nan
string_classes = (str, bytes)
int_classes = int
import collections.abc

container_abcs = collections.abc
FileNotFoundError = FileNotFoundError


def load_bert_sentences(config):
    processor = DATASET_MAP[config.dataset]()
    config.num_labels = processor.NUM_CLASSES
    config.is_multilabel = processor.IS_MULTILABEL
    train_examples, dev_examples, test_examples = processor.get_sentences()

    train_texts, train_labels = [], []
    dev_texts, dev_labels = [], []
    test_texts, test_labels = [], []

    for example in train_examples:
        train_texts.append(example.text)
        train_labels.append(example.label)

    for example in dev_examples:
        dev_texts.append(example.text)
        dev_labels.append(example.label)

    for example in test_examples:
        test_texts.append(example.text)
        test_labels.append(example.label)


    train_dataset = Data(train_texts, train_labels)
    dev_dataset = Data(dev_texts, dev_labels)
    test_dataset = Data(test_texts, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.TEST.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.TEST.batch_size)

    config.num_labels = processor.NUM_CLASSES


    config.TRAIN.num_train_optimization_steps = int(
        len(
            train_examples) / config.TRAIN.batch_size / config.TRAIN.gradient_accumulation_steps) * config.TRAIN.max_epoch

    return train_dataloader, dev_dataloader, test_dataloader



def multi_acc(y, preds):
    preds = torch.argmax(torch.softmax(preds, dim=-1), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def multi_mse(y, preds):
    mse_loss = torch.nn.MSELoss()
    preds = torch.argmax(torch.softmax(preds, dim=-1), dim=1)
    return mse_loss(y.float(), preds.float())


def generate_over_tokenizer(text, tokenizer, max_length=256):
    t = tokenizer.batch_encode_plus(text, padding='max_length',
                                    max_length=max_length,
                                    truncation=True,
                                    )
    input_ids = torch.tensor(t["input_ids"])
    attention_mask = torch.tensor(t["attention_mask"])
    return input_ids, attention_mask



def save2file(data, path):
    data = data.tolist()
    f = open(path, "a+")
    for line in data:
        str_lin = ""
        for item in line:
            str_lin += str(item) + " "
        f.write(str_lin +"\n")
    f.close()


def load_pretrained_segment_adaembeddings(config, load_pretrained=True):
    try:
        if not load_pretrained:
            raise Exception
        train = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], "train_ada.pt"))
        dev = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], "dev_ada.pt"))
        test = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], "test_ada.pt"))
        print("===loading adaadaembeding over fine-tuned models...")
        print("Done!")
    except:
        print("===generating adaembeding over fine-tuned models...")
        device = 'cuda:3'

        from transformers import BertTokenizer, BertModel
        from datasets.dataset import generate_sents

        dataset = DATASET_MAP[config.dataset]()

        config.num_labels = dataset.NUM_CLASSES
        config.is_multilabel = dataset.IS_MULTILABEL

        # loading data
        d_train, d_dev, d_test = dataset.get_documents()

        pretrained_weights = os.path.join(SAVED_MODEL_PATH, config.dataset)
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        encoder = BertModeladptive.from_pretrained(pretrained_weights, num_labels=config.num_labels).eval().to(device)

        agent = torch.load(os.path.join(SAVED_MODEL_PATH, config.dataset) + '/agent.pkl')
        agent = agent.eval().to(device)
        #
        # if config.n_gpu > 1:
        #     encoder = torch.nn.DataParallel(encoder)

        print("== traning datasets")
        embeddings = []
        labels = []
        lengths = []
        for step, example in enumerate(d_train):
            sentences = generate_sents(example.text)
            with torch.no_grad():
                 t = tokenizer.batch_encode_plus(sentences, padding='max_length',
                                            max_length=250,
                                            truncation=True,
                                            )
                 input_ids = torch.tensor(t["input_ids"]).to(device)
                 attention_mask = torch.tensor(t["attention_mask"]).to(device)

                 pro = agent(input_ids)
                 action = gumbel_softmax(pro.view(pro.size(0), -1, 2))
                 policy = action[:, :, 1]
                 embedding = encoder(input_ids=input_ids, policy=policy,
                                attention_mask=attention_mask)[1]
            #if step ==50:
            #    break
            embeddings.append(embedding.cpu().detach())
            labels.append(example.label)
            lengths.append(len(sentences))
            print("\rIteration: {:>5}...".format(step), end="")
        data = embeddings, labels, lengths
        torch.save(data, os.path.join(DATASET_PATH_MAP[config.dataset], "train_ada.pt"))
        print("Done!".ljust(30))

        print("== dev datasets...")
        embeddings = []
        labels = []
        lengths = []

        for step, example in enumerate(d_dev):
            sentences = generate_sents(example.text)
            with torch.no_grad():

                 t = tokenizer.batch_encode_plus(sentences, padding='max_length',
                                            max_length=250,
                                            truncation=True,
                                            )
                 input_ids = torch.tensor(t["input_ids"]).to(device)
                 attention_mask = torch.tensor(t["attention_mask"]).to(device)

                 pro = agent(input_ids)
                 action = gumbel_softmax(pro.view(pro.size(0), -1, 2))
                 #if step == 50:
                 #    break
                 policy = action[:, :, 1]

                 embedding = encoder(input_ids=input_ids, policy=policy,
                                attention_mask=attention_mask)[1]

            embeddings.append(embedding.cpu().detach())
            labels.append(example.label)
            lengths.append(len(sentences))
            print("\rIteration: {:>5}...".format(step), end="")
        data = embeddings, labels, lengths

        torch.save(data, os.path.join(DATASET_PATH_MAP[config.dataset], "dev_ada.pt"))
        print("Done!".ljust(30))

        print("== test datasets...")
        embeddings = []
        labels = []
        lengths = []

        for step, example in enumerate(d_test):
            sentences = generate_sents(example.text)
            with torch.no_grad():

                 t = tokenizer.batch_encode_plus(sentences, padding='max_length',
                                            max_length=250,
                                            truncation=True,
                                            )
                 input_ids = torch.tensor(t["input_ids"]).to(device)
                 attention_mask = torch.tensor(t["attention_mask"]).to(device)

                 pro = agent(input_ids)
                 action = gumbel_softmax(pro.view(pro.size(0), -1, 2))
                 policy = action[:, :, 1]
                 #if step == 50:
                 #    break
                 embedding = encoder(input_ids=input_ids, policy=policy,
                                attention_mask=attention_mask)[1]

            embeddings.append(embedding.cpu().detach())

            labels.append(example.label)
            lengths.append(len(sentences))
            print("\rIteration: {:>5}...".format(step), end="")
        data = embeddings, labels, lengths

        torch.save(data, os.path.join(DATASET_PATH_MAP[config.dataset], "test_ada.pt"))
        print("Done!".ljust(30))

        train = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], "train_ada.pt"))
        dev = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], "dev_ada.pt"))
        test = torch.load(os.path.join(DATASET_PATH_MAP[config.dataset], "test_ada.pt"))

    return (train, dev, test)



def load_data_decoder(config, load_pretrained=True):
    train, dev, test = load_pretrained_segment_adaembeddings(config, load_pretrained)
    train_embeddings, train_labels, train_lengths= train
    train_dataset = Data(train_embeddings, train_labels, train_lengths)
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True,
                              collate_fn=default_collate)

    dev_embeddings, dev_labels, dev_lengths= dev
    dev_dataset = Data(dev_embeddings, dev_labels, dev_lengths)
    dev_loader = DataLoader(dev_dataset, batch_size=config.TRAIN.batch_size, shuffle=False, collate_fn=default_collate)

    test_embeddings, test_labels, test_lengths = test
    test_dataset = Data(test_embeddings, test_labels, test_lengths)
    test_loader = DataLoader(test_dataset, batch_size=config.TRAIN.batch_size, shuffle=False,
                             collate_fn=default_collate)


    config.num_labels = LABLES[config.dataset]
    return train_loader, dev_loader, test_loader



def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        max_length = 0
        for b in batch:
            if max_length < len(b): max_length = len(b)
        new_batch = []
        for b in batch:
            new_batch.append(torch.cat([b, torch.zeros(max_length - len(b), 768)]))
        out = None
        return torch.stack(new_batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
