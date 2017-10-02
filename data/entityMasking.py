from tqdm import tqdm


def masking(infile, outfile):
    fin = open(infile)
    fout = open(outfile, "w")
    print("Processing {}".format(infile))
    for line in tqdm(fin.readlines()):
        instanceid, sentence, label = line.strip().split('%%%%')
        sentence = sentence.strip().split()
        label = label.strip().split()
        if len(sentence) != len(label):
            print("Length mismatch in file : {}".format(infile))
        sen_str = []
        e_str = []
        flag = False
        for token, tag in zip(sentence, label):
            if token == '<pad>':
                break
            if tag == 'O':
                if flag:
                    flag = False
                sen_str.append(token)
            if tag == 'I':
                if flag == False:
                    sen_str.append('<e>')
                    flag = True
                e_str.append(token)
        if len(e_str) == 0:
            # We regard the whole sentence as entity here
            fout.write("{}\t{}\n".format(" ".join(sen_str), " ".join(sen_str)))
        else:
            fout.write("{}\t{}\n".format(" ".join(sen_str), " ".join(e_str)))








if __name__=="__main__":
    masking("main-train-results.txt", "entity_mask.train")
    masking("main-valid-results.txt", "entity_mask.valid")
    masking("main-test-results.txt", "entity_mask.test")