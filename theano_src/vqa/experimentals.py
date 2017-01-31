import numpy as np
import sys
sys.path.insert(0, '../pystruct/models')

from scipy.sparse import csr_matrix
from scipy import io
from scipy.sparse import vstack
from functions import *
import json
from collections import OrderedDict


best_accuracy = {0: 0}
def train(weights_vec, threshold, threshold_minimization, learning_rate, world_list_collection, train_Y, question_ids, object_vectors, vocab):
     #max_iteration
    # for index, word_list in questions_train.items():
    print(" at begining of training threshold: ", threshold)
    for index in range(question_ids.__len__()):
        # print(" #########################: ")
        if (index % 100 == 0): print("train", ":: current index: ", index, " threshold: ", threshold)
        q_id = question_ids[index]
        word_list = world_list_collection[q_id]
        objects_list = object_vectors[q_id]
        annotation = train_Y[index]
        # print(" index: ", index, " q_id: ", q_id, annotation, "\n", objects_list["0"])

        table = genearte_score_table(word_list, objects_list, weights_vec, vocab)
        h_fi_tuple = unconditional_argmax(word_list, dict(table), objects_list, vocab)
        # temp = 0
        # for word, object_id in h_fi_tuple["h"].items():
        #     temp+=table[word][object_id]
        # weight_vec_threshold_tupple = update_weight_vec_threshold(weights_vec, list(word_list.keys()), dict(table), objects_list, vocab, h_fi_tuple, annotation,threshold_minimization, learning_rate, threshold)


        weight_vec_threshold_tupple = update_weight_vec_threshold(weights_vec, list(word_list), dict(table), objects_list, vocab, h_fi_tuple, annotation,threshold_minimization, learning_rate, threshold)
        threshold = weight_vec_threshold_tupple["threshold"]
        weights_vec = weight_vec_threshold_tupple [ "weight_vec"]
        #break
    # print("#########################################################################")
    #weight_threshold_write_json_object = {"threshold": threshold, "weight": list(weights_vec.toarray().flatten())}
    weight_threshold_write_json_object = {"threshold": threshold, "weight": weights_vec}
    print("after training threshold is: ", threshold)
    #with open("model.txt", "w")as model_f:
        #print("writing: wait")
         #json.dump(dict(weight_threshold_write_json_object), model_f)
    return weight_threshold_write_json_object

def test_(weights_vec, threshold, threshold_minimization, learning_rate, world_list_collection, actual_val_Y, question_ids, object_vectors, vocab):
    correct = 0
    print(" #### validation with threshold: ", threshold)
    #with open("val_annaotations.txt", "w") as val_f:
    for index in range(question_ids.__len__()):
        # print(" #########################: ")
        if (index % 100 == 0): print("val", ":: current index: ", index, " so far correct: ", correct)
        q_id = question_ids[index]
        word_list = world_list_collection[q_id]
        objects_list = object_vectors[q_id]
        #annotation = train_Y[index]
        # print(" index: ", index, " q_id: ", q_id, annotation, "\n", objects_list["0"])

        table = genearte_score_table(word_list, objects_list, weights_vec, vocab)
        h_fi_tuple = unconditional_argmax(word_list, dict(table), objects_list, vocab)
        predict_ans = binary_inference_given_threshold(h_fi_tuple["h"], table, threshold)
        #if(predict_ans==1): val_f.write("yes")
        #else: val_f.write("no")
        if(actual_val_Y[index]==predict_ans): correct+=1
            # index+=1
            #break
    accuracy = correct / actual_val_Y.__len__()
    #print("correct: ", correct, " of: ", actual_val_Y.__len__(), " percentage: ", accuracy, " question_ids: ", question_ids.__len__(), " annotation: ", actual_val_Y.__len__())
    print("correct: ", correct, " of: ", actual_val_Y.__len__(), " percentage: ", accuracy)
    return accuracy
SETS = ["train", "val"]

def routine(weights_vec, threshold, threshold_minimization, learning_rate, train_Y, actual_val_Y, vocab):
    global best_accuracy
    max_iter_per_superstep = 20
    same_counter = 0
    previous_accuracy = 0
    previous_threshold = 0
    for i in range(max_iter_per_superstep):
        print(" iteration: ", i, " #########################: ")
        if(i>0):
            if(weights_vec.toarray().flatten().all()!=weights_vec_previous.toarray().flatten().all()):
                print(" got difrrent weight vec than preciously trained iteration")
        for set_name in SETS:
            print(set_name, " ############")
            file_name = "../Binary_" + set_name + "_abstract_questions/logical_formed_binary_question_ids_" + set_name + "_2015.txt"
            question_id_file = "../Binary_" + set_name + "_abstract_questions/binary_question_ids_" + set_name + "_2015.txt"
            world_list_collection = {}
    

            if(set_name=="train"):
                weight_threshold_tuple = train(weights_vec, threshold, threshold_minimization, learning_rate, world_list_collection, train_Y, question_ids,object_vectors, vocab)
                weights_vec = weight_threshold_tuple["weight"]
                weights_vec_previous = weight_threshold_tuple["weight"]
                threshold = weight_threshold_tuple["threshold"]
                #print("threshold: ", threshold)
                #print("after training weight vec: ", weights_vec)
                if(weights_vec.toarray().flatten().all()!= weight_threshold_tuple["weight"].toarray().flatten().all()):
                    print("##################something wrong###############")

            else:
                #print(" threshold: ", threshold)
                #print("weight vec: ", weights_vec)
                accuracy = test_(weights_vec, threshold, threshold_minimization, learning_rate, world_list_collection, actual_val_Y, question_ids, object_vectors, vocab)
        print("for iteration: ", i, " accuracy: ", accuracy, " so far was best : ",best_accuracy[0])
        if(accuracy>best_accuracy[0]):
            max_iter_per_superstep+=1
            print(" it is better than so far best: so updating and writing and increaseing max iteration by 1")
            max_iter_per_superstep+=1
            best_accuracy [0] = accuracy
            score_f = "best_weight_vec_"+str(best_accuracy[0])+"_thresold_"+str(threshold)+"_rate_"+str(learning_rate) + ".mtx"
            io.mmwrite(score_f, weights_vec)
            best_accuracy["threshold"] = threshold
            best_accuracy["threshold_minimization"] = threshold_minimization
            best_accuracy["file"] = score_f
            with open ("best_par.json", "w") as best_f:
                json.dump(best_accuracy, best_f)
        if((accuracy - previous_accuracy)<0.00001):
            same_counter += 1
            print("almost same or worse accuracy as previous for ", same_counter, " times.")
            #thresholda_minimization *=  (1+same_counter)
        else:
            previous_accuracy = accuracy
            same_counter = 0
        #if((threshold-previous_threshold)<0.1):
            #learning_rate /= 2
        if(same_counter>=5):
            print("same accuracy for 3 times. should quit")
            return accuracy

    return accuracy

    #
with open("binaryVocab.txt", "r") as bin_vocab_f:
    vocab = json.load(bin_vocab_f) #2792 dstinct words
    # vocab = ["a", "b", "c"]

train_Y = getAnnotations("../Annotations_train_abstract_v002/binary_train")
actual_val_Y = getAnnotations("../Annotations_val_abstract_v002/binary_val")

for xx in range(3):
    print(" Superstep: ", xx,"###########################################################################")
    weights_vec = initialize_weight_vec(vocab.__len__() * 564,-1)  # initialize by all 1 and in csr_matrix
    score_f = "superstep_%s_initial.txt" % xx + ".mtx"
    io.mmwrite(score_f, weights_vec)
    threshold = 25
    threshold_minimization = 0.01
    learning_rate = 0.01
    accuracy = routine(weights_vec, threshold, threshold_minimization, learning_rate, train_Y, actual_val_Y, vocab)
    if(accuracy >=.8):
        print("###################################################################")
        break

    #report_f.write(str(accuracy))

