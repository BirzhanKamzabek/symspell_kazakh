import os
from pathlib import Path
import pandas as pd
from symspellpy import SymSpell, Verbosity
from kaznlp.normalization.ininorm import Normalizer

from kaznlp.tokenization.tokrex import TokenizeRex
from kaznlp.tokenization.tokhmm import TokenizerHMM

from kaznlp.lid.lidnb import LidNB

MAX_EDIT_DISTANCE = 2
DICTIONARY_PATH = Path('kk.txt')

symspell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE)
symspell.load_dictionary(DICTIONARY_PATH, term_index=0, count_index=1, encoding='utf-8')

def lookup_file(path):
    p = path.split('.')
    extns = p[len(p) - 1]
    if extns == 'json':
        df = pd.read_json(path)
    elif extns == 'csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_table(path)

    data = {'original':[], 'corrected':[]}

    for row in df.values:
        data['original'].append(row)
        correct_msg = ""
        for word in row[0].split(" "):
            correct_word = lookup_word(word, Verbosity.TOP, MAX_EDIT_DISTANCE)
            if correct_word != None:
                correct_msg = correct_msg + " " + correct_word
            else:
                correct_msg = correct_msg + " " + word
        data['corrected'].append(correct_msg)

    df = pd.DataFrame(data)
    df.to_csv('./out.csv')

def lookup_word(word, verbosity, max_edit_distance):
    suggestions = symspell.lookup(word, verbosity, max_edit_distance)
    words_list = [item.term for item in suggestions]

    if verbosity == Verbosity.CLOSEST:
        return words_list

    if len(words_list) > 0:
        return words_list[0]

    return None

def lang_detector(text):
    tokrex = TokenizeRex()
    landetector = LidNB(char_mdl=os.path.join('kaznlp', 'lid', 'char.mdl'))
    print()
    doclan = landetector.predict(tokrex.tokenize(text, lower=True)[0])
    print(f'Document "{text}" is written in {doclan}.')

    print()
    doclan = landetector.predict_wp(tokrex.tokenize(text, lower=True)[0])
    print(f'Document "{text}" has the following language probabilities {doclan}.')

    print()

def correction_text(txt):
    mdl = os.path.join('kaznlp', 'tokenization', 'tokhmm.mdl')
    tokhmm = TokenizerHMM(model=mdl)
    sents_toks = tokhmm.tokenize(txt)

    for word in sents_toks[0]:
        print(lookup_word(word, Verbosity.TOP, MAX_EDIT_DISTANCE)+" ", end="")

    print()


def main():

    while True:
        print("Your method scanning:")
        print("1. Input value")
        print("2. File path")
        n = int(input())
        if n == 1:
            print('Enter your text:')
            txt = input()

            lang_detector(txt)

            correction_text(txt)

            break
        elif n == 2:
            print('Enter your file path:')
            path = input()

            lookup_file(path)

            break
        else:
            print('---Choose valid number of option---')

if __name__ == "__main__":
    main()