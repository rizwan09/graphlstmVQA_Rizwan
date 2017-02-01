import os
import json

DATA_PATH="vqa_data"
BATCH_SIZE = 8



#LEARNING_RATE_VARIATIONS = [0.005,  0.01, 0.05]
LEARNING_RATE_VARIATIONS = [ 0.01]
NEPOCHS_VARIATIONS = [50]
RATE_VARIATIONS = [0.03]
L2REG_REG_WEIGHT_VARIATIONS = [0.0]
DECAY_EPOCHS_VARIATIONS = [5]
MINIMUM_LR_VARIATIONS = [1e-5]
MLP_HIDDEN_OUT_DIM_VARIATIONS = [10000]

total = LEARNING_RATE_VARIATIONS.__len__() * RATE_VARIATIONS.__len__() * RATE_VARIATIONS.__len__() * DECAY_EPOCHS_VARIATIONS.__len__() * MLP_HIDDEN_OUT_DIM_VARIATIONS.__len__()*\
    NEPOCHS_VARIATIONS.__len__()*MINIMUM_LR_VARIATIONS.__len__()*L2REG_REG_WEIGHT_VARIATIONS.__len__()

i = 0
all_settings =[]
for lr in LEARNING_RATE_VARIATIONS:
    for rate_1 in RATE_VARIATIONS:
        emb_dropout_rate =  rate_1
        for rate_2 in RATE_VARIATIONS:
            lstm_dropout_rate = rate_2
            for decay_epochs in DECAY_EPOCHS_VARIATIONS:
                for MLP_hidden_out_dim in MLP_HIDDEN_OUT_DIM_VARIATIONS:
                    for nepochs in NEPOCHS_VARIATIONS:
                        for minimum_lr in MINIMUM_LR_VARIATIONS:
                            for L2Reg_reg_weight in L2REG_REG_WEIGHT_VARIATIONS:
                                settings = {"lr:": lr, "emb_dropout_rate": emb_dropout_rate,\
                                            "lstm_dropout_rate": lstm_dropout_rate,\
                                            "MLP_hidden_out_dim": MLP_hidden_out_dim,\
                                            "minimum_lr": minimum_lr, 'L2Reg_reg_weight': L2Reg_reg_weight,\
                                            'nepochs': nepochs, "decay_epochs": decay_epochs,\
                                            'current': i, 'total': total}
                                all_settings.append(settings)
                                print' ########################\nstarting superstep ', i+1, ' of  ', total
                                run_command = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python2.7 theano_src/lstm_VQA.py --train_path ' + DATA_PATH + '/train_ --valid_path ' + DATA_PATH + \
                                          '/val_ --featurelist_path ' + DATA_PATH + '/featureList.txt --emb_dir /if1/kc2wc/data/glove/glove.6B.300d_w_header.txt --batch_size '+str(BATCH_SIZE)+' --lr ' + str(lr) + ' --emb_dropout_rate ' + str(emb_dropout_rate) + \
                                              ' --lstm_dropout_rate ' + str(lstm_dropout_rate) + ' --circuit VQA --decay_epochs '+str(decay_epochs)+' --MLP_hidden_out_dim '+ str(MLP_hidden_out_dim) +' --current '+str(i+1)+ ' --total '+str(total)+\
                                              ' --L2Reg_reg_weight '+ str(L2Reg_reg_weight)+' --minimum_lr '+ str(minimum_lr)+ ' --nepochs '+ str(nepochs)+ ' --label_path '+DATA_PATH+ '| tee Object_vqa_results'

                                os.system(run_command)
                                #print 'finished lr: ', lr, ' emb drobout rate: ', emb_dropout_rate, ' lstm drop out rate: ', lstm_dropout_rate
                                i+=1
                                with open("done_record.txt", "w")as done_f:
                                    done_f.write(str(i)+" of "+str(total))
                                print ' ########################finished superstep ', i ,' of  ',total
