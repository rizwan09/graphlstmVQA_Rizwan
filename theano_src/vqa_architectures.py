import theano.tensor as T
from neural_lib import *

def calculate_params_needed(chips):
    l = []
    for c, _ in chips:
        l += c.needed_key()
    return l

''' Automatically create(initialize) layers and hook them together'''
def stackLayers(chips, current_q_chip, current_obj_chip, params, feature_size=0, entity_size=2):
    instantiated_chips = []
    current_chip = None
    question_chip = object_chip = None
    print 'stack layers!!!'
    for e in chips:
        if 'question_emb' in e[1]:
            current_chip = current_q_chip
        elif 'object_emb' in e[1]:
            current_chip = current_obj_chip
        previous_chip = current_chip
        if e[1].endswith('feature_emission_trans'):
            current_chip = e[0](e[1], params).prepend(previous_chip, feature_size)
        elif e[1].endswith('target_columns'):
            current_chip = e[0](e[1], params).prepend(previous_chip, entity_size)
        elif e[1].endswith('attention'):
            current_chip = e[0](e[1], params).prepend(question_chip, object_chip)
        else:
            current_chip = e[0](e[1], params).prepend(previous_chip)
        if 'question' in e[1]:
            question_chip = current_chip
        elif 'object' in e[1]:
            object_chip = current_chip

        instantiated_chips.append((current_chip, e[1]))
        #print 'current chip:', e[1], "In_dim:", current_chip.in_dim, "Out_dim:", current_chip.out_dim
        print 'needed keys:'
        for e in current_chip.needed_key():
            print (e, params[e])
    return instantiated_chips 

''' Compute the initialized layers by feed in the inputs. '''
def computeLayers(instantiated_chips, current_q_chip, current_obj_chip, params, feature_input=None,  maskq=None, maskobj=None):
    print 'compute layers!!!'
    trainable_parameters = []
    object_chip = None
    question_chip = None
    for e in instantiated_chips:
        if 'question_emb' in e[1]:
            current_chip = current_q_chip
        elif 'object_emb' in e[1]:
            current_chip = current_obj_chip 
        previous_chip = current_chip
        current_chip = e[0]
        #print 'current chip:', e[1], "In_dim:", current_chip.in_dim, "Out_dim:", current_chip.out_dim
        if e[1].endswith('feature_emission_trans'):
            internal_params = current_chip.parameters
            current_chip.compute(previous_chip.output_tv, feature_input)
        elif e[1].endswith('lstm'):
            internal_params = current_chip.parameters
            if 'question' in e[1]:
                current_chip.compute(previous_chip.output_tv, maskq)
            else:
                current_chip.compute(previous_chip.output_tv, maskobj)
        elif e[1].endswith('attention'):
            internal_params = current_chip.parameters
            current_chip.compute(question_chip.output_tv, object_chip.output_tv, maskq, maskobj)
        else:
            internal_params = current_chip.parameters
            current_chip.compute(previous_chip.output_tv)
        assert current_chip.output_tv is not None
        for k in internal_params:
            print 'internal_params:', k.name
            assert k.is_trainable
            params[k.name] = k
            trainable_parameters.append(k)

        if 'question' in e[1]:
            question_chip = current_chip
        elif 'object' in e[1]:
            object_chip = current_chip

    return trainable_parameters 

''' Compile the architectures for single-task learning: stack the layers, compute the forward pass.'''
def VQAStackMaker(chips, params, graph=False, weighted=False, batched=False):
    
    assert 'oemb_matrix' in params or 'qemb_matrix' in params
    if batched:
        emb_q_inputs = T.itensor3('emb_q_input')
        emb_obj_inputs = T.itensor3('emb_obj_input')
        if graph:
            if weighted:
                masks_q = T.ftensor4('child_mask')
                masks_obj = T.ftensor4('child_mask')
            else:
                masks_q = T.ftensor3('child_mask')
                masks_obj = T.ftensor3('child_mask')
        else:
            masks_q = T.fmatrix('batch_mask')
            masks_obj = T.fmatrix('batch_mask')
    else:
        emb_q_inputs = T.imatrix('emb_q_input') 
        emb_obj_inputs = T.imatrix('emb_obj_input') 
        if graph:
            if weighted:
                masks_q = T.ftensor3('child_mask')
                masks_obj = T.ftensor3('child_mask')
            else:
                masks_q = T.fmatrix('child_mask')
                masks_obj = T.fmatrix('child_mask')
        else:
            masks_q = masks_obj = None
    #print masks, type(masks), masks.ndim
    current_q_chip = Start(params['q_voc_size'], emb_q_inputs)  
    current_obj_chip = Start(params['o_voc_size'], emb_obj_inputs)  

    
    print '\n', 'Building Stack now', '\n', 'Start: ', params['q_voc_size'] #, 'out_tv size:', len(current_chip.output_tv)
    instantiated_chips = stackLayers(chips, current_q_chip, current_obj_chip, params, entity_size=params['num_entity'])
    trainable_parameters = computeLayers(instantiated_chips, current_q_chip, current_obj_chip, params,  maskq=masks_q, maskobj=masks_obj)

    current_chip = instantiated_chips[-1][0]
    if current_chip.output_tv.ndim == 2:
        pred_y = current_chip.output_tv #T.argmax(current_chip.output_tv, axis=1)
    else:
        pred_y = current_chip.output_tv #T.argmax(current_chip.output_tv) #, axis=1)
    gold_y = (current_chip.gold_y
            if hasattr(current_chip, 'gold_y')
            else None)
    # Show all parameters that would be needed in this system
    params_needed = calculate_params_needed(instantiated_chips)
    print "Parameters Needed", params_needed
    for k in params_needed:
        assert k in params, k
        print k, params[k]
    assert hasattr(current_chip, 'score')
    cost = current_chip.score #/ params['nsentences'] 
    cost_arr = [cost]
    for layer in instantiated_chips[:-1]:
        if hasattr(layer[0], 'score'):
            print layer[1]
            cost += params['cost_coef'] * layer[0].score
            cost_arr.append(params['cost_coef'] * layer[0].score)

    grads = T.grad(cost,
            wrt=trainable_parameters)
    print 'trainable parameters:'
    print trainable_parameters
    if graph or batched:
        #return (emb_inputs, masks, entities_tv, attention_weights, entity_tvs, gold_y, pred_y, cost, grads, trainable_parameters) 
        return (emb_q_inputs, emb_obj_inputs, masks_q, masks_obj, gold_y, pred_y, cost, grads, trainable_parameters) 
    else: 
        return (emb_q_inputs, emb_obj_inputs, gold_y, pred_y, cost, grads, trainable_parameters) 

''' Single task architectures, suitable for CNN, BiLSTM (with/without input attention). 
The major difference between this and the graphLSTM single task architecture is the final line: the paremeters given to VQAStackMaker function'''

def VQA(params):
    chips = [
            (Embedding          ,'question_emb'),
#            (BiLSTM             ,'question_lstm'),
            (Embedding          ,'object_emb'),
            (Question_Object_attention,'attention'),
#        (MeanPooling        ,'pooling'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg')
            ]
    return VQAStackMaker(chips, params, batched=(params['batch_size']>1))

def Question(params):
    chips = [
            (Embedding          ,'emb'),
    #        (BiLSTM             ,'lstm'),
            (MeanPooling        ,'pooling'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg')
            ]
    return VQAStackMaker(chips, params, batched=(params['batch_size']>1))

def Object(params):
    chips = [
            (Embedding          ,'emb'),
            (MeanPooling        ,'pooling'),
            (LogitRegression    ,'logistic_regression'),
            (L2Reg,             'L2Reg')
            ]
    return VQAStackMaker(chips, params, batched=(params['batch_size']>1))


#=========================================
def LSTMRelation(params):
    chips = [
            #(Multi_Embeddings          ,'emb'),
            (Embedding          ,'emb'),
            (BiLSTM               ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            #(Entity_attention   ,'hidden_Entity_Att'),
            #(BiasedLinear       ,'MLP_hidden'),
            #(Activation         ,'MLP_activation'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return VQAStackMaker(chips, params, batched=(params['batch_size']>1))




def LSTMRelation(params):
    chips = [
            #(Multi_Embeddings          ,'emb'),
            (Embedding          ,'emb'),
            (BiLSTM               ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            #(Entity_attention   ,'hidden_Entity_Att'),
            #(BiasedLinear       ,'MLP_hidden'),
            #(Activation         ,'MLP_activation'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return VQAStackMaker(chips, params, batched=(params['batch_size']>1))

def CNNRelation(params):
    chips = [
            (Embedding          ,'emb'),
            (Entity_attention   ,'input_Entity_Att'),
            (Convolutional_NN   ,'CNN'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return VQAStackMaker(chips, params, batched=(params['batch_size']>1))



def GraphLSTMRelation(params):
    chips = [
            (Multi_Embeddings          ,'emb'),
            (BiGraphLSTM  ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return VQAStackMaker(chips, params, graph=True, batched=(params['batch_size']>1))


def WeightedGraphLSTMRelation(params):
    chips = [
            (Multi_Embeddings          ,'emb'),
            (BiGraphLSTM_Wtd  ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return VQAStackMaker(chips, params, graph=True, weighted=True, batched=(params['batch_size']>1))


def WeightedAddGraphLSTMRelation(params):
    chips = [
            (Embedding          ,'emb'),
            #(BiGraphLSTM_WtdAdd  ,'lstm'),
            (BiGraphLSTM_WtdEmbMult  ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return VQAStackMaker(chips, params, graph=True, weighted=True, batched=(params['batch_size']>1))


def ArcPredAddGraphLSTMRelation(params):
    chips = [
            (Embedding          ,'emb'),
            (BiGraphLSTM_ArcPred  ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            (LogitRegression    ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return VQAStackMaker(chips, params, graph=True, weighted=True, batched=(params['batch_size']>1))


''' Multitask learning architectures'''
def LSTMRelation_multitask(params, num_tasks):
    Shared = [
            (Embedding          ,'emb'),
            (BiLSTM               ,'lstm'),
            ]
    Classifiers = [[
            (TargetHidden       ,'t'+str(i)+'_get_target_columns'),
            (LogitRegression        ,'t'+str(i)+'_logistic_regression')
            ] for i in range(num_tasks)]
    return MultitaskVQAStackMaker(Shared, Classifiers, params, num_tasks, batched=(params['batch_size']>1))


def WeightedGraphLSTMRelation_multitask(params, num_tasks):
    Shared = [
            (Embedding          ,'emb'),
            (BiGraphLSTM_Wtd  ,'lstm'),
            ]
    Classifiers = [[
            (TargetHidden       ,'t'+str(i)+'_get_target_columns'),
            (LogitRegression        ,'t'+str(i)+'_logistic_regression')
            ] for i in range(num_tasks)]
    return MultitaskVQAStackMaker(Shared, Classifiers, params, num_tasks, graph=True, weighted=True, batched=(params['batch_size']>1))


def WeightedAddGraphLSTMRelation_multitask(params, num_tasks):
    Shared = [
            (Embedding          ,'emb'),
            (BiGraphLSTM_WtdAdd  ,'lstm'),
            ]
    Classifiers = [[
            (TargetHidden       ,'t'+str(i)+'_get_target_columns'),
            (LogitRegression        ,'t'+str(i)+'_logistic_regression')
            ] for i in range(num_tasks)]
    return MultitaskVQAStackMaker(Shared, Classifiers, params, num_tasks, graph=True, weighted=True, batched=(params['batch_size']>1))

