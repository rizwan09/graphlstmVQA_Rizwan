import matplotlib.pyplot as plt
import json
with open('all_settings.json', "r") as all_set_f:
    all_settings = json.load(all_set_f)
    #print(all_settings.__len__())
with open('best_parameters.json', "r") as all_set_f:
    best_parameters = json.load(all_set_f)
    #print(best_parameters['vf1'])
with open('best_scores_per_setting.json', "r") as all_set_f:
    best_score_per_setting = json.load(all_set_f)
    #print(best_score_per_setting.__len__())
with open('graph_data_train.json', "r") as all_set_f:
    graph_data_train = json.load(all_set_f)
    #print(graph_data_train[0].__len__())
with open('graph_data_valid.json', "r") as all_set_f:
    graph_data_valid= json.load(all_set_f)
    #print(graph_data_valid[0].__len__())

for i in range (all_settings.__len__()):
    if(i==2 or i==0):
        setting = all_settings[i]
        #print('set: ', i, " lr: ", setting['lr:'], "L2 Reg weight: ", setting['L2Reg_reg_weight'], " minimum lr: " , setting['minimum_lr'])
        train = graph_data_train[i]
        valid = graph_data_valid[i]
        plt.plot(train, label='set:' + str(i))
        #plt.plot(valid, label='set:' + str(i))
        print("best validation score for set: ", i, " train: ", best_score_per_setting[i]['trainf1'], " valid: ", best_score_per_setting[i]['vf1'])

#plt.plot([0.6783230, 0.682293330, 0.880000], label='firstline')
#plt.plot( [0.67555, 00.876545, 0.76], label='2ndline')
plt.ylabel('valid accuracy where each setting has\n both emb and lstm dropout 0.4,\n decay epoch 50, mlp hidden 1000')
plt.legend()
plt.show()