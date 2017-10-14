import numpy as np
import os
import re
import pickle as pkl
from console_logging.console import Console
console = Console()

'''
Preprocessing:
remove everything except lettes spaces exclamations question marks @symbol

Features:
one hot encoded words
one hot encoded capital words (if no capitals, 0)
count of exlamation (!) and question mark (?)
Later: one hot encoded mentions (@username)
'''

# Debugging
console.setVerbosity(4)
# Training
# console.setVerbosity(3)
# Staging
# console.setVerbosity(2)
# Production
# console.mute()
# Neater logging inside VS Code
console.timeless()
console.monotone()

DATASET_FILEPATH = 'data/text_emotion.csv'
dataset_path = os.path.join(os.getcwd(), DATASET_FILEPATH)
console.log("Loading data from %s" % dataset_path)


def _clean(sentence):
    regex_letters = "a-zA-Z"
    regex_spaces = " "
    regex_symbols = "!?@&;.,"
    regex_pattern = regex_letters + regex_spaces + regex_symbols
    new_sentence = re.sub('[^%s]' % regex_pattern, '', sentence)
    regex_special_characters = "&.*?;"
    regex_punctuation = "[.,]"
    return re.sub(regex_punctuation,'',re.sub(regex_special_characters, '', new_sentence))


def load_data():
    lines = open(dataset_path, 'r', encoding='utf8').readlines()
    data = [{'emotion': line.split(',')[1][1:-1],
             'raw': _clean(','.join(line.split(',')[3:]))} for line in lines]
    return data


data_save_path = os.path.join(os.getcwd(), 'data/data.sav')
if os.path.exists(data_save_path):
    console.log("Reading from save file...")
    data = pkl.load(open(data_save_path, 'rb'))
    console.success("Finished reading data from save.")
else:
    console.log("Did not find a save file.")
    data = load_data()
    pkl.dump(data, open(data_save_path, 'wb'))
    console.success("Created save file.")

console.info("First data is sentence \"%s\" with emotion \'%s\'" %
             (data[0]['raw'], data[0]['emotion']))


def make_wordlists(data):
    wordlist = set()
    mentions = set()
    uppercase = set()
    for datapoint in data:
        words = re.sub('[ ]{1,10}', ',', datapoint['raw'])
        words = re.sub('[?!]', '', words).split(',')
        for word in words:
            if len(word) > 0:
                if word[0] == '@':
                    mentions.add(word[1:])
                else:
                    if word.isupper():
                        uppercase.add(word.lower())
                    wordlist.add(word.lower())
    wordlist = np.asarray(list(wordlist))
    mentions = np.asarray(list(mentions))
    return mentions, wordlist


wordlist_path = os.path.join(os.getcwd(), 'data/wordlist.npy')
mentions_path = os.path.join(os.getcwd(), "data/mentions.npy")
if os.path.exists(wordlist_path):
    console.log("Reading from existing wordlist...")
    wordlist = np.load(wordlist_path)
    mentions = np.load(mentions_path)
    console.success("Finished importing wordlist.")
else:
    console.log("Did not find an existing wordlist.")
    mentions, wordlist = make_wordlists(data)
    np.save(wordlist_path, wordlist)
    np.save(mentions_path, mentions)
    console.success("Created and saved a new wordlist.")

console.info("There are %d words in wordlist." % len(wordlist))
console.info("First 10 words are %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" %
             tuple(wordlist[:10]))
