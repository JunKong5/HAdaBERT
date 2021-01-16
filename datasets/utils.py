import re
import sys
import csv
import torch
import torch.utils.data


# csv.field_size_limit(sys.maxsize)


class InputExample(object):
    def __init__(self, guid=None, text=None, label=None):
        self.guid = guid
        self.text = text
        self.label = label



class Data(torch.utils.data.Dataset):
    sort_key = None
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)



class SentenceProcessor(object):
    NAME = 'SENTENCE'



    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
                # print(line)

            return lines












