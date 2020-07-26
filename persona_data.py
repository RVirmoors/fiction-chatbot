# inspired by https://github.com/eleanorstrib/parse_dialog/blob/master/code.py

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def step_one_read_in(textfile):
    text = ''
    with open(textfile, 'rt', encoding="utf8") as file_in:
        for line in file_in:
            text = text + line
    return text


def preprocess(string):
    """
    Preprocessing:
    - Spaces before periods at end of sentences
    - everything lowercase
    """
    return string


def read_and_parse(textfile):
    persona = {
        "personality": ["exegesis .", "the female man ."],
        "utterances": []
    }
    persona["utterances"].append({
        "candidates": ["yes ."],
        "history": ["is there a god ?"]
    })

    parsed_dialog = []
    parsed_narrative = []

    questions = []  # list of Q & A pairs
    answers = []  # dialog statements with no corresponding Q

    quote_open, quote_close = '``', "''"
    with open(textfile, 'rt', encoding="utf8") as file_in:
        open_q = False
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
                        if open_q:
                            questions[-1]['a'] = cont_diag + current
                            open_q = False
                        if current[-1] == '?':
                            questions.append(
                                {'q': (cont_diag + current), 'a': []})
                            open_q = True
                        else:
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
                        if open_q and current[-1] in '.!':
                            questions[-1]['a'] = parsed_narrative[-1]
                            open_q = False
                        if current[-1] == '?':
                            questions.append(
                                {'q': parsed_narrative[-1], 'a': []})
                            open_q = True
                        else:
                            answers.append(parsed_narrative[-1])
                    current = []
                else:
                    current.append(word)
                counter += 1
                if current and not quotes:
                    if open_q and current[-1] in '.!':
                        questions[-1]['a'] = current
                        current = []
                        open_q = False
                    elif current[-1] == '?':
                        questions.append(
                            {'q': current, 'a': []})
                        open_q = True
                        current = []
            if current:
                # end of paragraph:
                answers.append(current)
    print(questions[15:20])
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


def write_persona(persona):
    print(persona)
    return persona


if __name__ == "__main__":
    #text = step_one_read_in('text1.txt')
    for i in range(1):
        persona = read_and_parse('text' + str(i + 1) + '.txt')
        # write_persona(persona)
        """
        print('Writing text #', i + 1)
        write_data(dialog, 'dial-' + str(i))
        write_data(narrative, 'narr-' + str(i))
        """
