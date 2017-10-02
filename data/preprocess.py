######################
# Create ID->Str dictionary : One-to-One mapping
#####################

import re
from collections import defaultdict
from fuzzywuzzy import process
import logging
from nltk.tokenize.treebank import TreebankWordTokenizer
from tqdm import tqdm
logger = logging.getLogger()
logger.disabled = True

idtostr = defaultdict(list)

tokenizer = TreebankWordTokenizer()

def extract(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str


# Read ID-Name from file
with open("FB5M.name.txt") as f:
    for line in tqdm(f.readlines()):
        line = line.split('\t')
        key, value = line[0], line[2]
        key = key[1:-1]
        value = value[1:-1]
        idtostr[key].append(value)

exact_match = False # Priority for Exact Match
count_exact_match = 0
total = 0

for item in ["train.txt", "valid.txt", "test.txt"]:
    fout = open("dictionary_"+item, "w")
    flog = open("logging_"+item, "w")
    with open("SimpleQuestions_v2/annotated_fb_data_"+item) as f:
        print("Processing {} file".format(item))
        for line_num, line in tqdm(enumerate(f.readlines())):
            total += 1
            exact_match = False
            line = line.split('\t')
            key, sent_ori = line[0], line[3]
            sent = tokenizer.tokenize(sent_ori) # Tokenize the sentence
            key = extract(key)
            try:
                candi = idtostr.get(key, [])
                if candi == []:
                    continue
                # Else, Use the whole sentence to select one entity, which has highest similarity with sentence
                v, score = process.extractOne(" ".join(sent), candi)
            except:
                print(line_num, line, sent, idtostr.get(key, []))
                exit()
            #value = re.findall(r"\w+|[^\w\s]", v, re.UNICODE)
            #fout.write("{}\t{}\n".format(key," ".join(value)))
            fout.write("{}\t{}\n".format(key, v))
    print("total = ",total)

