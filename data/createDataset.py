from tqdm import tqdm
from collections import defaultdict
id2str = defaultdict(str)


def extract(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def toToken(in_str):
    return " ".join(in_str.strip()[3:].replace('_','.').split('.'))

def lookup(subject):
    result = id2str[extract(subject)]
    if len(result) == 0:
        print("Can not find subject in dictionary: {}".format(subject))
        return "NOSUB"
    return result

def process(filename, goldfile, entityfile, outfile, ansfile):
    fgold = open(goldfile)
    gold_fact = []
    for line in fgold.readlines():
        line = line.strip().split('\t')
        if len(line) != 4:
            print("Line length error {}".format(line))
            exit(1)
        gold_fact.append([extract(line[0]), extract(line[1]), line[3]])
    print("There are {} instances in {}".format(len(gold_fact), goldfile))
    fentity = open(entityfile)
    mask_entity = []
    for line in fentity.readlines():
        line = line.strip().split('\t')
        if len(line) != 2:
            print("error {}".format(line))
            exit(0)
        mask_entity.append([line[0], line[1]])
    fin = open(filename)
    fout = open(outfile, "w")
    fans = open(ansfile, "w")
    flog = open("IDmatchSTRmis.log"+filename, "w")
    for line in fin.readlines():
        line = line.strip().split("%%%%")
        lineid = int(line[0].strip().split('-')[1])-1
        gold_subject = gold_fact[lineid][0]
        gold_subject_str = " ".join(list(lookup(gold_subject).replace(' ','^')))
        gold_relation = gold_fact[lineid][1]
        sentence = mask_entity[lineid][0]
        entity = " ".join(list(mask_entity[lineid][1].replace(' ','^')))

        if len(line) <= 1:
            print("{} There is no relation {}".format(filename, line))
        for instance in line[1:]:
            label = "0"
            instance = instance.strip().split('\t')
            linking_subject = instance[0]
            linking_str = " ".join(list(instance[1].replace(' ','^')))
            linking_relation = instance[2]
            linking_score = instance[3]
            if linking_subject == gold_subject and linking_relation == gold_relation:
                label = "1"
                if linking_str != gold_subject_str:
                    #print("Id Match but the string did not match")
                    flog.write("sub {} rel {} linking_str {} gold_str {}".format(gold_subject, gold_relation, linking_str, gold_subject_str))
            fout.write("\t".join([str(lineid), sentence, entity, linking_str, toToken(linking_relation), linking_score, label])+"\t"\
                     +"\t".join([str(lineid), sentence, entity, linking_str, toToken(linking_relation), linking_score, label])+"\n")
        fans.write("\t".join([str(lineid), sentence, gold_subject_str, toToken(gold_relation)])+"\n")

def process_pair(filename, goldfile, entityfile, outfile):
    fgold = open(goldfile)
    gold_fact = []
    for line in fgold.readlines():
        line = line.strip().split('\t')
        if len(line) != 4:
            print("Line length error {}".format(line))
            exit(1)
        gold_fact.append([extract(line[0]), extract(line[1]), line[3]])
    print("There are {} instances in {}".format(len(gold_fact), goldfile))
    fentity = open(entityfile)
    mask_entity = []
    for line in fentity.readlines():
        line = line.strip().split('\t')
        if len(line) != 2:
            print("error {}".format(line))
            exit(0)
        mask_entity.append([line[0], line[1]])
    fin = open(filename)
    fout = open(outfile, "w")
    for line in fin.readlines():
        line = line.strip().split("%%%%")
        lineid = int(line[0].strip().split('-')[1])-1
        gold_subject = gold_fact[lineid][0]
        gold_relation = gold_fact[lineid][1]
        sentence = mask_entity[lineid][0]
        entity = " ".join(list(mask_entity[lineid][1].replace(' ','^')))

        if len(line) <= 1:
            continue
        positive = []
        negative = []
        for instance in line[1:]:
            label = "0"
            instance = instance.strip().split('\t')
            linking_subject = instance[0]
            linking_str = " ".join(list(instance[1].replace(' ','^')))
            linking_relation = instance[2]
            linking_score = instance[3]
            if linking_subject == gold_subject and linking_relation == gold_relation:
                label = "1"
                positive.append("\t".join([str(lineid), sentence, entity, linking_str, toToken(linking_relation), linking_score, label]))
            else:
                negative.append("\t".join([str(lineid), sentence, entity, linking_str, toToken(linking_relation), linking_score, label]))
        if len(positive) == 0:
            continue
        else:
            for pos in positive:
                for neg in negative:
                    fout.write(pos+"\t"+neg+"\n")

def readLookupTable(filename):
    for file in filename:
        fin = open(file)
        print("Reading Dict")
        for line in tqdm(fin.readlines()):
            line = line.strip().split('\t')
            id2str[line[0]] = line[1]


if __name__=="__main__":
    readLookupTable(["dictionary_train.txt","dictionary_valid.txt","dictionary_test.txt"])
    process_pair("train-append.txt",
            "SimpleQuestions_v2/annotated_fb_data_train.txt",
            "entity_mask.train",
            "attentivecnn.train")
    process("valid-append.txt",
            "SimpleQuestions_v2/annotated_fb_data_valid.txt",
            "entity_mask.valid",
            "attentivecnn.valid",
            "attentivecnn.valid.gold")
    process("test-append.txt",
            "SimpleQuestions_v2/annotated_fb_data_test.txt",
            "entity_mask.test",
            "attentivecnn.test",
            "attentivecnn.test.gold")