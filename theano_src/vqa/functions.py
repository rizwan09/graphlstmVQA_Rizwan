import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import rand
from scipy import io
from scipy.sparse import vstack
def initialize_weight_vec(dim, value):
    # print(" in initialization ", dim, value, rand(1,dim, density=0.2).toarray())
    if(value ==0): return csr_matrix(np.zeros(dim))
    if(value==1): return csr_matrix(np.ones(dim))
    else:
        vec = rand(1,dim, density=0.01)
        #print("initializes superstep with: ", vec)
        return  vec


def fi(given_word, object, vocab):
    word_vec = csr_matrix(([1], ([0], [vocab.index(given_word)])), shape=(1, vocab.__len__())) #sparse 1 hot encoding
    # print(given_word, object,  "outer prod: ", np.outer(word_vec.toarray(),object).flatten())
    # return np.outer(word_vec,object).flatten()
    return  csr_matrix(np.outer(word_vec.toarray(),object).flatten())

def binary_inference_given_threshold(h , table_two_d, threshold):
    if(h.__len__()!=table_two_d.__len__()): return 0 #incomplete alignment
    sum_score = 0
    for word, object_id in h.items():
        sum_score +=table_two_d[word][object_id]
    if(sum_score>=threshold): return 1
    else: return 0

def get_fi_from_h(h, objects_list, vocab):
    fi_ = np.zeros(vocab.__len__() * objects_list["0"].__len__())
    for word, object_id in h.items():
        if(object_id!=-1): fi_ = np.add(fi_, fi(word, objects_list[object_id], vocab).toarray().flatten())
    return fi_ #not yet in csr matrix

def get_max_three_d(a):
    return max([(x, y, a[x][y]) for x in a for y in a[x]], key=lambda x: x[2])

def unconditional_argmax(word_list, table_two_d, objects_list, vocab):
    tuple = {}
    tuple["h"] = {}
    # tuple["fi"] = np.zeros(vocab.__len__()*objects_list[0].__len__())
    tuple["fi"] = np.zeros(vocab.__len__()*objects_list["0"].__len__())
    for count in range(word_list.__len__()):
        maximum = get_max_three_d(table_two_d)
        tuple["h"][maximum[0]] = maximum[1]
        tuple["fi"] = np.add(tuple["fi"], fi(maximum[0], objects_list[maximum[1]], vocab).toarray().flatten())
        table_two_d.pop(maximum[0], None)
        # for k, v in table_two_d.items():
        #     v.pop(maximum[1], None)
    return tuple

def get_total(table_two_d, unconstrained_h):
    sum_ = 0
    for word, object_id in unconstrained_h.items():
        sum_+=table_two_d[word][object_id]
    return sum_

def get_revised_table(table_two_d, threshold):
    # print("in recised table generation: ", threshold)
    closest_word_nodes = {}
    for word, scores in table_two_d.items():
        scores[-1] = 0
        closest_word_nodes[word] = sorted(scores, key = scores.__getitem__, reverse =True)
        max_level = closest_word_nodes[word].__len__()
        # print(word, closest_word_nodes[word], " max level or num objects: ", max_level)
    cut_off = False
    # threshold = 6.5
    # print("threshold: ", threshold)
    for count in range(max_level):
        if(cut_off != False):
            for word in table_two_d:
                # print("poping ", word,"'s node connection with node id: ", closest_word_nodes[word][count])
                table_two_d[word].pop(closest_word_nodes[word][count], None)
        else:
            sum_ = 0
            for word in table_two_d:
                # print(" level: ", count, "sum_: ", sum_, "word: ", word , " map to object id: ", closest_word_nodes[word][count], " score: ", table_two_d[word][closest_word_nodes[word][count]], " new sum: ", sum_+ table_two_d[word][closest_word_nodes[word][count]] )
                sum_ += table_two_d[word][closest_word_nodes[word][count]]
            if(sum_ < threshold):
                count += 1
                # print( threshold, " cut off level: ", count)
                cut_off = True

    # for word, scores in table_two_d.items():
    #     print("new table's :", word, scores)
    return table_two_d



def traverse(word_list, table_two_d_revised, h, previous_val, vocab, best_h_fi_score_tuple, threshold):
    # print(word_list, table_two_d_revised, h, previous_val, vocab, best_score, threshold)
    if(word_list.__len__()==0):
        tuple = {"h": h, "score":previous_val}
        # print("******* word list empty returning: ", tuple)
        return dict(tuple)
    else:
        start_word = word_list[0]
        # print(" got wordlist: ", word_list)
        word_list.remove(start_word)
        for object_id, val in table_two_d_revised[start_word].items():
            new_h = dict(h)
            new_h[start_word]= object_id
            new_val = previous_val+ val
            # print("##### ###### level: ", new_h.__len__(), " previous: ", h,  "starting from: ", start_word, "with: ", object_id, " new: ", new_h, " score: ", val, " new total: ", new_val)
            temp_h_score_tuple = traverse(list(word_list), table_two_d_revised, new_h, new_val, vocab, best_h_fi_score_tuple, threshold)
            # print("current: ", temp_h_score_tuple)
            # print("so far best: ", best_h_fi_score_tuple)
            if(temp_h_score_tuple["score"] < threshold and temp_h_score_tuple["score"]>best_h_fi_score_tuple["score"]):
                # print("entered updating ", temp_h_score_tuple["score"], " best: ", best_h_fi_score_tuple["score"])
                best_h_fi_score_tuple = dict(temp_h_score_tuple)
                # print("found new best: ", best_h_fi_score_tuple)
    return dict(best_h_fi_score_tuple)
def constrained_argmax(word_list, table_two_d, vocab, objects_list, threshold):
    tuple = {}
    tuple["h"] = {}
    tuple["fi"] = np.zeros(vocab.__len__() * objects_list["0"].__len__())  # outer prod initialization
    table_two_d_revised = get_revised_table(dict(table_two_d), threshold)
    # print("revised: ", table_two_d_revised)
    start_word = word_list[0]
    word_list.remove(start_word)
    best_h_fi_score_tuple ={}
    best_h_fi_score_tuple["h"] =  dict()
    best_h_fi_score_tuple["score"] =  0
    for object_id, val in table_two_d_revised[start_word].items():
        # print("##### ###### ####### strting from: ", start_word, " with: ", object_id, " score: ", val, " word list: ", word_list)
        h = {start_word: object_id}
        temp_h_score_tuple = traverse(list(word_list), table_two_d_revised, h, val, vocab, best_h_fi_score_tuple, threshold)
        if (temp_h_score_tuple["score"] < threshold and temp_h_score_tuple["score"] > best_h_fi_score_tuple["score"]):
            # print("entered updating ", temp_h_score_tuple["score"], " best: ", best_h_fi_score_tuple["score"])
            best_h_fi_score_tuple = dict(temp_h_score_tuple)
            # print("found new best at starting : ", best_h_fi_score_tuple)
    best_h_fi_score_tuple["fi"] = get_fi_from_h(best_h_fi_score_tuple["h"], objects_list, vocab)
    # print("returning h^ ", best_h_fi_score_tuple)
    return best_h_fi_score_tuple

def update_weight_vec_threshold(weight_vec, word_list, table_two_d, objects_list, vocab, unconstrained_h_fi_tuple, y, threshold_minimization, learning_rate, threshold):
    # print("h predicts: ", binary_inference_given_threshold(unconstrained_h_fi_tuple["h"], table_two_d, threshold))
    # print("threshold: ",threshold)
    tuple = {}
    y_predict = binary_inference_given_threshold(unconstrained_h_fi_tuple["h"], table_two_d, threshold)
    # if(y_predict==y):
        # print("no update needed")
    if(y_predict == 0 and y == 1):
        threshold -= threshold_minimization
        # print("new threshold: ", threshold)
    if(y_predict == 1 and y == 0):
        h_cap_fi_tuple = constrained_argmax(word_list, table_two_d, vocab, objects_list, threshold)
        # print("current weight vec: ",weight_vec.toarray().flatten())
        # print(" h: ", unconstrained_h_fi_tuple["h"])
        # print(" h fi: ", unconstrained_h_fi_tuple["fi"])
        # print(" h^cap: ", h_cap_fi_tuple["h"])
        # print(" h^cap fi: ", h_cap_fi_tuple["fi"])
        diff_vec = np.subtract(h_cap_fi_tuple["fi"], unconstrained_h_fi_tuple["fi"])
        #print("diff: ", csr_matrix(diff_vec))
        print(csr_matrix(h_cap_fi_tuple["fi"]))
        update_vec = np.dot(learning_rate, diff_vec)
        weight_vec = csr_matrix(np.add(weight_vec.toarray(), update_vec))
        # print(" updating weight vec: ", weight_vec.toarray())
    tuple["threshold"] = threshold
    tuple["weight_vec"] = weight_vec
    # print(" updated weight vec: ", tuple["threshold"], tuple["weight_vec"].toarray())

    return tuple

def genearte_score_table(word_list, objects_list, weights_vec, vocab):
    table = {}
    # print("in table geneartation:: weight vec: ", weights_vec.toarray() )
    for word in word_list:
        temp_table = {}
        for object_id, object_vec in objects_list.items():
            temp_table[object_id] = np.dot(weights_vec.toarray().flatten(), fi(word, object_vec, vocab).toarray().flatten())
        table[word] = temp_table
    return table

def best_alignment(word, available_node_ids, alignment_score):
    best_id = -1
    best_score = -99999999
    for id in available_node_ids:
        if(alignment_score[word][id]>best_score):
            best_score = alignment_score[word][id]
            best_id = id
    return best_id

