# from collections import OrderedDict
# VALUES = {}
# VALUES["null"] = -1000000 #just to represent numericallly but it should also
#
# class Object:
#     def __init__(self, id, features={}):
#         self.id = id
#         self.features = OrderedDict.fromkeys("")
#         for key in features:
#             self.features[key] = VALUES["null"]
#     def getFeatureVector(self):
#         vector_ = []
#         for key, val in self.features.items():
#             vector_.append(val)
#             return vector_
#     def _print(self):
#         print("Printing object. id: ", self.id, " feature vec size: ", self.features.__len__())
#         for key, val in self.features.items():
#             if(val!= VALUES["null"]):
#                 # if(key=="name"):
#                 print(key, val)


from collections import OrderedDict
VALUES = {}
VALUES["null"] = -1000000 #just to represent numericallly but it should also
RESET=0
SET=1
class Object:
    def __init__(self, id, choice, features={}):
        self.id = id
        self.features = OrderedDict.fromkeys("")
        for key in features:
            if(choice==RESET): self.features[key] = 0
            else: self.features[key] = features[key]

    def getFeatureVector(self):
        vector_ = []
        for key, val in self.features.items():
            vector_.append(val)
        return vector_

    def setFeatureVector(self, vector_ = []):
        index = 0
        for key in self.features:
            self.features[key]=vector_[index]
            index+=1
    def addFeature(self, key, val):
        if (isinstance(val, str) or key in ("deformable", "flip", "poseID", "typeID", "expressionID", "instanceID" )):
            key += "_"
            key += str(val)
            # print("Adding: ", key)
            val = 1
        if (isinstance(val, bool)):
            if(val==True): val = 1
            else: val = 0

        self.features[key] = val
        if (self.features.__len__()>564): print("Error: key not found: ", key)
    def _print(self):
        print("Printing object. id: ", self.id, " feature vec size: ", self.features.__len__())
        for key, val in self.features.items():
            if(val!= VALUES["null"]):
                # if(key=="name"):
                print(key, val)

    def _printVector(self):
        print(self.features)

        # i = 0
        # for key, val in self.features.items():
        #     print(key, val)
        #     i+=1
        #     if(i==10):break
