# inspired by https://github.com/eleanorstrib/parse_dialog/blob/master/code.py

import argparse
import queue
import sys

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import random
import re

import numpy as np

import json

# parse command line args
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)  # show docstring from top

parser.add_argument(
    '--files', type=int, default=3,
    help='# of text files to process.')

parser.add_argument(
    '--lines', type=int, default=6,
    help='# of sentences per row.')

args = parser.parse_args()


def step_one_read_in(textfile):
    text = ''
    with open(textfile, 'rt', encoding="utf8") as file_in:
        for line in file_in:
            text = text + line
    return text


def preprocess(tokenized):
    """
    Preprocessing:
    - detokenize
    - Spaces before periods at end of sentences
    - everything lowercase
    """
    s = TreebankWordDetokenizer().detokenize(tokenized)
    return s


def read_and_parse(textfile):
    sentences = []  # list of tokens
    data_lines = []  # list of lists of sentences: [answer, question, previous lines]

    quote_open, quote_close = '``', "''"
    with open(textfile, 'rt', encoding="utf8") as file_in:
        for line in file_in:
            quotes = []
            current = []
            cont_narr = False
            cont_diag = []
            tokenized = word_tokenize(line)
            counter = 0
            while counter < len(tokenized):
                word = tokenized[counter]
                if quotes and quote_close in word:  # ending quote
                    quotes.pop()
                    if current[-1] == ',':
                        cont_diag += current
                    # long quote = dialog
                    elif len(current) > 2 and current[-1] in ".!?":
                        sentences.append(cont_diag + current)
                        cont_diag = []
 #                   elif sentences and not cont_diag:   # short quote = narrative
                        #                        sentences[-1] += current
                        #                        print(sentences[-1])
                        #                        cont_narr = True
                    current = []
                elif quote_open in word:
                    quotes.append(quote_open)     # starting quote
                    if len(current) > 3:
                        if cont_narr:
                            sentences[-1] += current
                            cont_narr = False
                        elif current[0].lower() not in "saidsays":
#                            print("QO:", current[0])
                            sentences.append(current)
                            cont_narr = False
                    current = []
                else:
                    current.append(word)
                counter += 1
                if current and not quotes and len(current) > 3:
                    # regular text, outside quotes
                    if current[-1] in ".!?":
                        if current[0].lower() not in "saidsays":
#                            print("RT:", current[0])
                            sentences.append(current)
                        current = []
            if current and current[-1] in ".!?" and len(current) > 3:
                # end of paragraph:
                if current[0].lower() not in "saidsays":
#                    print("EP:", current[0])
                    sentences.append(current)
                current = []

        for i in range(len(sentences) - 1):
            data_line = ['', '', '', '', '', '']
            data_line[0] = preprocess(sentences[i + 1])     # response
            data_line[1] = preprocess(sentences[i])         # context
            data_line[2:args.lines] = [preprocess(sentences[j])  # four other random lines
                                       for j in random.sample(range(len(sentences)), args.lines - 2)]
            data_lines.append(data_line)
    return data_lines


def write_dialog(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(filename, "written.")


if __name__ == "__main__":
    dataset = []
    for i in range(args.files):
        data = read_and_parse('text' + str(i + 1) + '.txt')
        print("Extracted lines from text", i + 1)
        dataset += data
    write_dialog(dataset, 'dialog_data.json')
