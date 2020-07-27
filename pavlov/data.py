# inspired by https://github.com/eleanorstrib/parse_dialog/blob/master/code.py

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def step_one_read_in(textfile):
    text = ''
    with open(textfile, 'rt', encoding="utf8") as file_in:
        for line in file_in:
            text = text + line
    return text


def read_and_parse(textfile):
    parsed_dialog = []
    parsed_narrative = []
    bunch_dialog = []       # 50 lines at a time
    bunch_narrative = []    # 50 lines at a time
    # current text (can be dialog or narrative)
    quote_open, quote_close = '``', "''"
    with open(textfile, 'rt', encoding="utf8") as file_in:
        for line in file_in:
            quotes = []
            current = []
            cont_narr = False
            tokenized = word_tokenize(line)
            counter = 0
            while counter < len(tokenized):
                word = tokenized[counter]
                if quotes and quote_close in word:
                    quotes.pop()
                    if len(current) > 3:  # long quote = dialog
                        parsed_dialog.append(current)
                        if len(parsed_dialog) >= 50:
                            bunch_dialog.append(parsed_dialog)
                            parsed_dialog = []
                    elif parsed_narrative:   # short quote = narrative
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
                        if len(parsed_narrative) >= 50:
                            bunch_narrative.append(parsed_narrative)
                            parsed_narrative = []
                            cont_narr = False
                    current = []
                else:
                    current.append(word)
                counter += 1
    return bunch_dialog, bunch_narrative


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


def write_persona(bunch):
    print(dialog[0])
    persona = []
    return persona


if __name__ == "__main__":
    #text = step_one_read_in('text1.txt')
    for i in range(2):
        dialog, narrative = read_and_parse('text' + str(i + 1) + '.txt')
        persona = write_persona(dialog)
        """
        print('Writing text #', i + 1)
        write_data(dialog, 'dial-' + str(i))
        write_data(narrative, 'narr-' + str(i))
        """
