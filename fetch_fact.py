from collections import defaultdict
import random
from tqdm import tqdm

random.seed(2324)

kb = defaultdict(list)
id2str = defaultdict(str)

def readFB(filename):
    print("Reading file from 2M Freebase")
    fin = open(filename)
    lineid = 0
    for line in tqdm(fin.readlines()):
        linesplit = line.strip().split('\t')
        lineid += 1
        if len(linesplit) == 3:
            subject, predicate, object = linesplit[0], linesplit[1], linesplit[2]
            if (predicate, object) not in kb[subject]:
                kb[subject].append((predicate, object))
        else:
            print(lineid, len(linesplit), line, end='')

def constructDataset(dataset, dataname):
    print("Reading file from", dataset)
    fin = open(dataset)
    fout = open(dataname, "w")
    for line in tqdm(fin.readlines()):
        linesplit = line.strip().split("%%%%")
        lineid = linesplit[0].strip()
        result = [lineid]
        if len(linesplit) > 1:
            for item in linesplit[1:]:
                itemsplit = item.strip().split('\t')
                sub = itemsplit[0]
                sub_str = itemsplit[1]
                score = itemsplit[2]
                obj_collection = kb[sub]
                for obj in obj_collection:
                    result.append("\t".join([sub, sub_str, extract(obj[0]), score]))
        fout.write(" %%%% ".join(result)+"\n")

def extract(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def lookup(subject):
    result = id2str[extract(subject)]
    if len(result) == 0:
        print("Can not find subject in dictionary: {}".format(subject))
        return "NOSUB"
    return result

def constructDatasetWithSample(dataset, dataname, golddata):
    fgold = open(golddata)
    gold = []
    print("Reading data from ", golddata)
    for line_id, line in enumerate(fgold.readlines()):
        linesplit = line.strip().split('\t')
        if len(linesplit) != 4:
            print("Error in line {}".format(line_id))
            exit()
        subject = extract(linesplit[0])
        relation = extract(linesplit[1])
        object = extract(linesplit[2])
        gold.append([subject, relation, object])
    print(len(gold))
    fin = open(dataset)
    fout = open(dataname, "w")
    print("Reading data from", dataset)
    flog = open("train_creatingdataset.log", "w")
    fsr = open("subject_relation_object.log", "w")
    for line in fin.readlines():
        linesplit = line.strip().split("%%%%")
        lineid = linesplit[0].strip()
        lid = int(lineid.split('-')[1])
        gold_fact = gold[lid-1]
        samples = []
        result = []
        if len(linesplit) > 1:
            for item in linesplit[1:]:
                itemsplit = item.strip().split('\t')
                sub = itemsplit[0]
                sub_str = itemsplit[1]
                score = itemsplit[2]
                obj_collection = kb[sub]
                for obj in obj_collection:
                    rel = extract(obj[0])
                    buffer = "\t".join([sub, sub_str, rel, score])
                    if sub == gold_fact[0] and rel == gold_fact[1]:

                        if buffer not in samples:
                            samples.append(buffer)

                    else:
                        if buffer not in result:
                            result.append(buffer)
        # incorporate the gold file here
        if len(samples) == 0:
            flog.write("{} Cannot find Gold Ans in Linking Fact Pool | The subject Entity is {}\n".format(lineid, sub, sub_str, ))
            samples.append("\t".join([gold_fact[0], lookup(gold_fact[0]), gold_fact[1], "1.0"]))
        if len(result) > 99:
            samples += random.sample(result, 99)
        else:
            samples += result
        samples = [lineid] + samples
        fout.write(" %%%% ".join(samples)+"\n")




def readLookupTable(filename):
    fin = open(filename)
    print("Reading Dict")
    for line in tqdm(fin.readlines()):
        line = line.strip().split('\t')
        id2str[line[0]] = line[1]






if __name__=="__main__":
    readFB("data/fb-2M-augmented.txt")
    readLookupTable("data/dictionary_train.txt")
    constructDatasetWithSample("data/train-h100.txt", "data/train-append.txt", "data/SimpleQuestions_v2/annotated_fb_data_train.txt")
    constructDataset("data/valid-h100.txt", "data/valid-append.txt")
    constructDataset("data/test-h100.txt", "data/test-append.txt")