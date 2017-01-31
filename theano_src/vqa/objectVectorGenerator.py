import numpy as np
import pickle
from collections import OrderedDict
import json
from Object import *

SETS_NAME = {
    "Train": "train"
    ,
    "Val": "val"
}
RESET=0
SET=1
#initialiging annotaions data structure
features_list = OrderedDict.fromkeys("")




with open("featureList.txt", "r")as feature_f:
    index = 0
    for line in feature_f:
        for word in line.split(","):
            if(word==""):continue
            features_list[word] = index
            index+=1

# for key in features_list:
#     print("key: ",key)
#store annoation table
annotations = {}

for set_name in SETS_NAME:
    print(set_name)
    image_composoition = {}
    images = {}
    object_vectors = OrderedDict.fromkeys("")
    annotations[set_name] = OrderedDict.fromkeys("")
    file_name = "../Binary_"+SETS_NAME[set_name]+"_abstract_questions/binary_question_ids_"+SETS_NAME[set_name]+"_2015.json"
    scene_root = "../scene_json_abstract_v002_"+SETS_NAME[set_name]+"2015/"
    image_file = scene_root+"abstract_v002_"+SETS_NAME[set_name]+"2015_scene_information.json"
    scene_folder = scene_root + "scene_composition_abstract_v002_"+SETS_NAME[set_name]+"2015/"
    # print(file_name)
    with open(file_name, "r") as annotation_f,\
        open(image_file, "r") as image_f:
        data = []
        data = json.load(image_f)
        for composition in data["compositions"]:
            image_composoition[composition["image_id"]] = composition["file_name"]
        data = []
        data = json.load(annotation_f)
        for answers in data:
            # if (answers["question_id"] ==2): print(answers)
            image_id = answers["image_id"]
            annotations[set_name][answers["question_id"]] = answers["multiple_choice_answer"]
            images[answers["question_id"]] = scene_folder+image_composoition[image_id]
            # print("In loading data: ", set_name,  answers["question_id"])



    objects_list = OrderedDict.fromkeys("")
    i = 0
    for question_id, scene_file in images.items():
        if(i%500==0):print(set_name, "gathering image info  iteration: ", i)
        i+=1
        objects_list[question_id] = []
        with open(scene_file, "r") as scene_f:
            data = json.load(scene_f)
            availableObject = data["scene"]["availableObject"]
            for object in availableObject:
                for ins in object["instance"]:
                    if(ins["present"]==True): objects_list[question_id].append(ins)
        # break
    i=0
    for question_id, scene_file in images.items():
        if (i % 500 == 0):print(set_name, "extracting image iteration: ", i)
        i+=1
        objects = OrderedDict.fromkeys("")
        index = -1
        for object in objects_list[question_id]:
            index = index + 1
            # print("object: ",object)
            obj = Object(index, RESET, features_list)
            for key, val in object.items():
                if(isinstance(val, list)):
                    if(key=="body"): #for other list processing is not necessary
                        for item in val:
                            part_name = item["part"]
                            temp = key + "_" + part_name + "_"
                            index_ = object["partIdxList"][part_name]
                            # obj.features[temp + "deformableX"] = object["deformableX"][index_]
                            # obj.features[temp + "deformableY"] = object["deformableY"][index_]
                            # obj.features[temp + "deformableLocalRot"] = object["deformableLocalRot"][index_]
                            # obj.features[temp + "deformableGlobalRot"] = object["deformableGlobalRot"][index_]

                            obj.addFeature(temp + "deformableX", object["deformableX"][index_])
                            obj.addFeature(temp + "deformableY", object["deformableY"][index_])
                            obj.addFeature(temp + "deformableLocalRot", object["deformableLocalRot"][index_])
                            obj.addFeature(temp + "deformableGlobalRot", object["deformableGlobalRot"][index_])

                            # print("item: ", item)
                            for module in item:
                                if(module!="part"):
                                    if(isinstance(item[module],list)):
                                        temp = key + "_" + item["part"] + "_" + module
                                        temp_index = 0
                                        for vv in item[module]:
                                            temp_name = temp+"_"+str(temp_index)
                                            # obj.features[temp_name] = item[module][temp_index]
                                            obj.addFeature(temp_name, item[module][temp_index])
                                            temp_index = temp_index + 1
                                    else:
                                        temp = key+ "_"+item["part"]+"_"+module
                                        # print("module: ", temp )
                                        # obj.features[temp] = item[module]
                                        obj.addFeature(temp, item[module])

                else:
                    if(key not in ("present","partIdxList")):
                        # obj.features[key] = val
                        obj.addFeature(key, val)
            # print(obj.getFeatureVector())
            objects[obj.id] = obj.getFeatureVector()
        object_vectors[question_id] = objects
        # break


    # for q_id in object_vectors:
        # print("Question_iD:", q_id)
        # for obj_id, vec in object_vectors[q_id].items():
        #     print(obj_id, vec)


    with open("objectVectors2_%s.txt" % SETS_NAME[set_name], "w") as obj_V_f:
        json.dump(object_vectors, obj_V_f)



    # with open("objectVectors_%s.txt" % SETS_NAME[set_name], "r") as obj_V_f:
    #     object_vectors = json.load(obj_V_f)
    #     for q_id in object_vectors:
    #         print("Question_iD:", q_id)
    #         for obj_id, vec in object_vectors[q_id].items():
    #             print(obj_id,  vec[features_list["type_human"]], vec[features_list["deformable_False"]], vec[features_list["deformable_True"]])

