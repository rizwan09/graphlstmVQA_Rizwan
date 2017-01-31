import json
import functools
def core_word(word):
    word = word.replace(',', '')
    word = word.replace("'", '')
    word = word.replace('[', '')
    word = word.replace("]", '')
    word = word.replace(")", '')
    word = word.replace("(", '')
    word = word.replace("?", '')
    return word


def getAnnotations(file):
    array_temp = []
    with open(file, "r") as train_annoattion_f:
        for line in train_annoattion_f:
            array_temp.append(line.strip())
            """
            if (line == "no\n"):
                array_temp.append(0)
            else:
                array_temp.append(1)
            """
    return array_temp

def read_question(file_name):
    question_txt = []
    with open(file_name, "r") as questions_f:
            for line in questions_f:
                line = [ core_word(word) for word in line.strip().split()]
                question_txt.append(line)
    return question_txt



def loadMulticlassProblem(args):
    train_ans = getAnnotations(args.train_a)
    val_ans = getAnnotations(args.val_a)
    train_q = read_question(args.train_q)
    val_q = read_question(args.val_q)
    # load obj
    train_obj = []
    val_obj = []
    objs = json.load(open(args.train_obj))
    featurelist = open(args.featurelist_path).readlines()[0].strip().split(',')
    for id in open(args.train_id).readlines():
        # now let's just sum up all feature vectors
        features = functools.reduce(lambda x,y:[x1+y1 for x1,y1 in zip(x,y)], objs[id.strip()].values())
        train_obj.append([ featurelist[idx].replace('name_','') for idx in range(len(featurelist)) \
            if features[idx] !=0 and 'name' in featurelist[idx]])
    objs = json.load(open(args.val_obj))
    for id in open(args.val_id).readlines():
        # now let's just sum up all feature vectors
        features = functools.reduce(lambda x,y:[x1+y1 for x1,y1 in zip(x,y)], objs[id.strip()].values())
        val_obj.append([ featurelist[idx].replace('name_','') for idx in range(len(featurelist)) \
            if features[idx] !=0 and 'name' in featurelist[idx]])

    print("train q[0]",  train_q[0])
    return (train_q, val_q, train_ans, val_ans, train_obj, val_obj)


def loadBinaryProblem(args):
    train_ans = getAnnotations(args.train_a)
    val_ans = getAnnotations(args.val_a)
    train_q = read_question(args.train_q) 
    val_q = read_question(args.val_q)
    # load obj
    train_obj = []
    val_obj = []
    objs = json.load(open(args.train_obj))
    featurelist = open(args.featurelist_path).readlines()[0].strip().split(',')
    for id in open(args.train_id).readlines():
        # now let's just sum up all feature vectors
        features = functools.reduce(lambda x,y:[x1+y1 for x1,y1 in zip(x,y)], objs[id.strip()].values())
        train_obj.append([ featurelist[idx].replace('name_','') for idx in range(len(featurelist)) \
            if features[idx] !=0 and 'name' in featurelist[idx]])
    objs = json.load(open(args.val_obj))
    for id in open(args.val_id).readlines():
        # now let's just sum up all feature vectors
        features = functools.reduce(lambda x,y:[x1+y1 for x1,y1 in zip(x,y)], objs[id.strip()].values())
        val_obj.append([ featurelist[idx].replace('name_','') for idx in range(len(featurelist)) \
            if features[idx] !=0 and 'name' in featurelist[idx]])

    print("train q[0]",  train_q[0])
    return (train_q, val_q, train_ans, val_ans, train_obj, val_obj)
