# inspired by https://github.com/eleanorstrib/parse_dialog/blob/master/code.py

import argparse
import queue
import sys

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import random
import re

import json

# parse command line args
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)  # show docstring from top

parser.add_argument(
    '--files', type=int, default=2,
    help='# of text files to process.')
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
    s = re.sub('([.,!?()])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    s = s.lower()
    return s


def read_and_parse(textfile):
    persona = {
        "personality": ["exegesis .", "the female man ."],
        "utterances": []
    }

    parsed_narrative = []

    questions = []  # list of Q & A pairs
    answers = []  # dialog statements with no corresponding Q

    quote_open, quote_close = '``', "''"
    with open(textfile, 'rt', encoding="utf8") as file_in:
        open_q = open_qq = False
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
                    elif len(current) > 2:    # long quote = dialog
                        if open_qq:
                            questions[-1]['a'] = cont_diag + current
                            open_qq = False
                        if current[-1] == '?':
                            questions.append(
                                {'q': (cont_diag + current), 'a': []})
                            open_qq = True
                        elif current[-1] not in ":;-—*":
                            answers.append(cont_diag + current)
                        cont_diag = []
                    elif parsed_narrative and not cont_diag:   # short quote = narrative
                        parsed_narrative[-1] += current
                        cont_narr = True
                    current = []
                elif quote_open in word:
                    quotes.append(quote_open)     # starting quote
                    if len(current) > 3:
                        if cont_narr:
                            parsed_narrative[-1] += current
                            cont_narr = False
                        else:
                            parsed_narrative.append(current)
                            cont_narr = False
                            """
                        if open_qq:
                            questions[-1]['a'] = parsed_narrative[-1]
                            open_qq = False
                            """
                        if current[-1] == '?':
                            questions.append(
                                {'q': parsed_narrative[-1], 'a': []})
                            open_qq = True
                    current = []
                else:
                    current.append(word)
                counter += 1
                if current and not quotes:
                    # regular text, outside quotes
                    if current[-1] in ".!":
                        if open_q:
                            questions[-1]['a'] = current
                            open_q = False
                        else:
                            answers.append(current)
                        current = []
                    elif current[-1] == '?':
                        if open_q:
                            questions[-1]['a'] = current
                        questions.append(
                            {'q': current, 'a': []})
                        open_q = open_qq = True
                        current = []
            if current and current[-1] not in ":;-—*":
                # end of paragraph:
                answers.append(current)

        for _, question in enumerate(questions):
            history = [preprocess(question['q'])]
            candidates = [preprocess(item)
                          for item in random.sample(answers,  10)]
            candidates.append(preprocess(question['a']))
            persona["utterances"].append(
                {"candidates": candidates, "history": history})
    return persona


def write_data(bunch, name):
    for i in range(len(bunch)):
        text = ''
        for line in range(len(bunch[i])):
            detok = TreebankWordDetokenizer().detokenize(bunch[i][line])
            text += detok + '\n'
        filename = 'data/' + name + '-' + '{:2d}'.format(i) + '.txt'
        with open(filename, 'w', encoding="utf8") as file_out:
            file_out.write(text)
            file_out.close()


def write_persona(personas, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    print(filename, "written.")


if __name__ == "__main__":
    personas = []
    for i in range(args.files):
        new_pers = read_and_parse('text' + str(i + 1) + '.txt')
        print("Built persona", i + 1)
        personas.append(new_pers)
    write_persona(personas, 'data.json')
