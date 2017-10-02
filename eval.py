from collections import defaultdict

def select_gold(filename):
    fin = open(filename)
    gold = defaultdict(tuple)
    for line in fin.readlines():
        line = line.strip().split('\t')
        gold[int(line[0])] = (tuple(line))
    return gold

def clean(data):
    clean_data = []
    for instance in data:
        sent = " ".join(list(filter(lambda a: a!='<pad>', instance[1])))
        rel = " ".join(list(filter(lambda a: a!='<pad>', instance[2])))
        mention = " ".join(list(filter(lambda a: a!='<pad>', instance[3])))
        candidate = " ".join(list(filter(lambda a: a!='<pad>', instance[4])))
        clean_data.append((instance[0], sent, rel, mention, candidate, instance[5]))
    return clean_data

def ranking(data):
    data = clean(data)
    instance = sorted(data, key=lambda x:(x[0], x[5]))
    #dump(instance, "ranking.data")
    id = instance[0][0]
    predicted = []
    best_result = instance[0]
    for ins in instance:
        if id != ins[0]:
            predicted.append(best_result)
            id = ins[0]
            best_result = ins
        else:
            if ins[5] > best_result[5]:
                best_result = ins
    predicted.append(best_result)
    return sorted(predicted, key=lambda x: x[0])

def find(instance, gold_data):
    qid = instance[0]
    result = gold_data[int(qid)]
    if len(result) == 0:
        print("Cannot find the item in gold data {}".format(instance))
        exit(1)
    else:
        return result

def compare(pred, gold):
    correct = 0
    for pre_instance in pred:
        gold_instance = find(pre_instance, gold)
        # if predicted_candidate == gold_entity and predicted_relation == gold_relation
        if pre_instance[4] == gold_instance[2] and pre_instance[2] == gold_instance[3]:
            correct += 1
    return correct

def dump(data, filename):
    fout = open(filename, "w")
    for item in data:
        fout.write("{0[0]}\t{0[1]}\t{0[2]}\t{0[3]}\t{0[4]}\t{0[5]}\n".format(item))


def evaluation(data_to_eval, gold_data_file):
    gold = select_gold(gold_data_file)
    pred = ranking(data_to_eval)
    dump(pred,"dump.data")
    correct = compare(pred, gold)
    acc = correct / len(gold)
    return acc