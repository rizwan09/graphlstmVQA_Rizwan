__doc__ = """Utilities for loading language datasets.

Basically porting http://github.com/stanfordnlp/treelstm/tree/master/util to Python.

"""

import tree_rnn

import numpy as np
import os
import codecs as cs
import sys

OOV = '_OOV_'


def read_RE_dataset_graph(data_dir, folds, dev_fold, num_entities=2, miniBatch=None):
    max_degree = 0
    data_folds = []
    train_data = []
    dev_data = []
    # load all the data
    for i in range(folds):
        folder = str(i)#.zfill(2)
    #for folder in ['train', 'dev']:
        subdir = os.path.join(data_dir, folder)
        dependencies, arc_type_dict = read_graph_dependencies(os.path.join(subdir, 'graph_arcs'))
        #dependencies = read_dependencies(os.path.join(subdir, 'dependency_labels'))
        sentences = read_sentences_RE(os.path.join(subdir, 'sentences_2nd'), num_entities)
        temp_max_degree = max([len(elem) for line in dependencies for elem in line])
        if temp_max_degree > max_degree:
            max_degree = temp_max_degree
        print 'graph max degree:', max_degree #, 'arc type size:', len(arc_type_dict)
        data_folds.append((dependencies, sentences))
    # get the vocabulary map and the label map.
    words = []
    for (dependencies, sentences) in data_folds:
        print len(dependencies), len(sentences)
        words.extend([w for sentence in sentences[0] for w in sentence])
    words2idx = {OOV: 0}
    for w in words:
        if w not in words2idx:
            words2idx[w] = len(words2idx)    
    print 'voc_size:', len(words2idx)
    print 'arc types:', arc_type_dict
    #for k, v in words2idx.iteritems():
    #    print k,v
    labels2idx = {'+':1, '-':0}
    dics = {'words2idx': words2idx, 'labels2idx': labels2idx}
    # convert words to indices
    for i, (dependencies, sentences) in enumerate(data_folds):
        sentences[0] = [[words2idx[w] for w in sent] for sent in sentences[0]]
        for j, (dependency, sentence, label, idx) in enumerate(zip(dependencies, *sentences)):
            try:
                assert np.all(np.array(sentence)>=0)
            except:
                print 'OOV word!!!!!!', sentence
            #_convert_dependencies(dependency, max_degree)
            #child_exist = np.zeros([len(dependency), len(dependency), 1]).astype('float32') 
            child_exist = np.zeros([len(dependency), len(dependency)]).astype('float32') 
            child_exist[0, 0] = 1
            for ii, elem in enumerate(dependency):
                #for jj, el in enumerate(elem):
                #child_exist[ii, elem[0], 0] = 1
                #child_exist[ii, elem[0]] = 1
                child_exist[ii, ii-1] = 1
            
            #child_exist = gen_child_mask_from_dep(dependency, len(arc_type_dict))
            try:
                assert len(child_exist) == len(sentence)
            except:
                print 'dep length does not match sent_len!!! fold', i, 'instance', j
                print len(child_exist), len(sentence), dependency, sentence 
            if i == dev_fold:
                dev_data.append((sentence, dependency, child_exist, label, idx))
            else:
                train_data.append((sentence, dependency, child_exist, label, idx))
    return train_data, dev_data, dics, max_degree, arc_type_dict

def gen_child_mask_from_dep(dependency, num_arc_type):
    child_exist = np.zeros([len(dependency), len(dependency), num_arc_type]).astype('float32') 
    for ii, elem in enumerate(dependency):
        if ii != 0:
            child_exist[ii, ii-1, 0] = 1
        for jj, el in enumerate(elem):
            child_exist[ii, el[0], el[1]] = 1
    return child_exist


def read_RE_dataset_tree(data_dir, folds, dev_fold, miniBatch=None):
    max_degree = 0
    data_folds = []
    train_data = []
    dev_data = []
    # load all the data
    for i in range(folds):
        folder = str(i) #.zfill(2)
    #for folder in ['train', 'dev']:
        subdir = os.path.join(data_dir, folder)
        trees = read_trees_no_labels(os.path.join(subdir, 'dlabels.chain'))#'dependency_labels'))#'dlabels.chain'))# #'dependency_labels'))
        sentences = read_sentences_RE(os.path.join(subdir, 'sentences'))#'sentences_no_entity'))
#'sentences'))#        
        temp_max_degree = max([degree for (_, degree) in trees])
        if temp_max_degree > max_degree:
            max_degree = temp_max_degree
        print 'tree max degree:', max_degree
        data_folds.append((trees, sentences))
    # get the vocabulary map and the label map.
    words = []
    for (trees, sentences) in data_folds:
        print len(trees), len(sentences)
        words.extend([w for sentence in sentences[0] for w in sentence])
    words2idx = {OOV: 0}
    for w in words:
        if w not in words2idx:
            words2idx[w] = len(words2idx)    
    print 'voc_size:', len(words2idx)
    #for k, v in words2idx.iteritems():
    #    print k,v
    labels2idx = {'+':1, '-':0}
    dics = {'words2idx': words2idx, 'labels2idx': labels2idx}
    # convert words to indices
    for i, (trees, sentences) in enumerate(data_folds):
        sentences[0] = [[words2idx[w] for w in sent] for sent in sentences[0]]
        for tree, sentence, label, idx in zip(trees, *sentences):
            try:
                assert np.all(np.array(sentence)>=0)
            except:
                print 'OOV word!!!!!!', sentence
            _remap_tokens_and_labels(tree[0], sentence, fine_grained=False)
            tree[0].label = label
            if i == dev_fold:
                dev_data.append((tree[0], label, idx))
            else:
                train_data.append((tree[0], label, idx))
    return train_data, dev_data, dics, max_degree


def convert_nn_input(dataset, max_degree):
    seqs = []
    labels = [] 
    deps = [] 
    idxs = []
    for root, label, idx in dataset:
        x, dependencies, _ = gen_nn_inputs(root, max_degree, only_leaves_have_vals=False,
                  with_labels=False)
        seqs.append(x)
        labels.append(label)
        deps.append(dependencies)
        idxs.append(idx)
    return seqs, labels, deps, idxs


def prepare_data(seqs, labels, deps, idxs):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence .
    
    param: seqs  a mini-batch of input seqences, array of array
    param: labels  a mini-batch of training labels, array of integers (booleans)
    param: deps  a mini-batch of topological dependencies for each node in each instance, array of matrices.
    param: idxs  a mini-batch of indices for each instance indicating where the entities located at, array of array of array.

    return: x, x_mask, labels, conv_deps, i, i_mask
    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    idx_lengths = [len(sid) for idx in idxs for sid in idx]

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    max_idx_len = numpy.max(idx_lengths)
    max_degree = len(deps[0][0])
    assert np.all(np.array([len(row) for item in deps for row in item]) == max_degree)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    i = numpy.zeros(max_idx_len, n_samples, len(idxs))
    i_mask = numpy.zeros((max_idx_len, n_samples, len(idxs))).astype(theano.config.floatX)
    new_deps = -1 * numpy.ones((maxlen, n_samples, max_degree))
    # Re-arrange the input-sequence, the dependency
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


# return a corpus x, y, idx
def read_sentences_RE(filename, num_entities=2):
    corpus_x = []
    corpus_y = []
    corpus_idx = []
    with cs.open(filename, 'r', encoding='utf-8') as inf:
        line_count = 0
        for line in inf:
            line_count += 1
            line = line.strip()
            if len(line) == 0:
                continue
            elems = line.split('\t') 
            entity_id_arry = []
            for ett in elems[1:1+num_entities]:
                entity_id = map(int, ett.split(' '))
                entity_id_arry.append(entity_id)
            assert len(entity_id_arry) == num_entities
            assert len(elems) == num_entities + 2
            x = elems[0].lower().split(' ')
            label = elems[-1]
            try:
                for i in range(num_entities):
                    assert entity_id_arry[i][-1] < len(x)
            except:
                sys.stderr.write('abnormal entity ids:'+str(entity_id_arry)+', sentence length:'+str(len(x))+'\n')
                continue
            #sentence = stringQ2B(sentence)
            if len(x) < 1:
                print x 
                continue
            #if label == '-':
            if label == 'None':
                y = 0
            else:
                y = 1
            corpus_x.append(x)
            corpus_y.append(y)
            corpus_idx.append(entity_id_arry)
    print 'read file', filename, len(corpus_x), len(corpus_y), len(corpus_idx)
    return [corpus_x, corpus_y, corpus_idx] 


def read_graph_dependencies(graph_file):
    dep_graphs = []
    arc_type_dict = dict()
    arc_type_dict['adjtok'] = 0
    with open(graph_file, 'r') as parents_f:
        while True:
            cur_parents = parents_f.readline()
            if not cur_parents :
                break
            cur_deps = [[elem.split('::') for elem in p.split(',,,')] for p in cur_parents.strip().split(' ')]
            for p in cur_parents.strip().split(' '):
                for elem in p.split(',,,'):
                    temp = elem.split('::')
                    try:
                        assert len(temp) ==2
                    except:
                        print elem, p
            dep_graphs.append(construct_graph_deps(cur_deps, arc_type_dict))
    return dep_graphs, arc_type_dict 

# get a dict of the arc_types and dependencies with types
def construct_graph_deps(dep_array, arc_type_dict):
    dep_graph = []
    for i, elem in enumerate(dep_array):
        local_dep = []
        if i == 0:
            local_dep.append((i, arc_type_dict['adjtok']))
        for pair in elem:
            arc_type = pair[0].split(':')[0]
            dep_node = int(pair[1])
            if dep_node < 0 or arc_type == 'prevsent' or arc_type == 'coref' or arc_type == 'discSenseInv' or arc_type == 'adjsent' or arc_type == 'depsent' or arc_type == 'depinv':   # 
                continue
            if dep_node < i:
                if arc_type not in arc_type_dict:
                    arc_type_dict[arc_type] = len(arc_type_dict) 
                local_dep.append((dep_node, arc_type_dict[arc_type]))
        try:
            assert (len(local_dep) > 0 or i == 0)
        except:
            print i, elem
        dep_graph.append(local_dep)
    return dep_graph

# Read the arc dependencies 
def read_dependencies(parents_file):
    dep_graphs = []
    with open(parents_file, 'r') as parents_f:
        while True:
            cur_parents = parents_f.readline()
            if not cur_parents :
                break
            cur_parents = [int(p) for p in cur_parents.strip().split()]
            dep_graphs.append(construct_dependency(cur_parents))
    return dep_graphs 

def construct_dependency(parent_list):
    dep_graph = [[i-1] for i in range(len(parent_list))]
    #dep_graph = [[] for i in range(len(parent_list))]
    dep_graph[0][0] = 0
    for i, elem in enumerate(parent_list):
        if elem == -1:
	    continue
	if i < elem and i not in dep_graph[elem]:
            dep_graph[elem].append(i)
        elif elem < i and elem not in dep_graph[i]:
            dep_graph[i].append(elem)
    #for i, elem in enumerate(dep_graph):
    #    if len(elem) == 0:
    #        elem.append(i-1)
    return dep_graph

# No labels on the tree arcs
def read_trees_no_labels(parents_file):
    trees = []
    with open(parents_file, 'r') as parents_f:
        line_num = 0
        while True:
            #print 'in line', line_num
            line_num += 1
            cur_parents = parents_f.readline()
            if not cur_parents :
                break
            cur_parents = [int(p) for p in cur_parents.strip().split()]
            trees.append(read_tree_no_labels(cur_parents))
    return trees

def read_tree_no_labels(parents):
    nodes = {}
    #parents = [p - 1 for p in parents]  # 1-indexed
    #print parents, len(parents)
    max_degree= 1
    for i in xrange(len(parents)):
        #print 'visit node:', i
        if i not in nodes:
            idx = i
            prev = None
            while True:
                #print 'read tree! idx:', idx
                node = tree_rnn.Node(val=idx, origin_idx=idx)  # for now, val is just idx
                if prev is not None:
                    assert prev.val != node.val
                    node.add_child(prev)

                nodes[idx] = node
                #print 'nodes:', nodes.keys()
                parent = parents[idx]
                if parent in nodes:
                    #assert len(nodes[parent].children) < 2
                    # Note: in our setting, a parent not nessacerily has only 2 children
                    nodes[parent].add_child(node)
                    if len(nodes[parent].children) > max_degree:
                        max_degree = len(nodes[parent].children)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    # ensure tree is completely binary
    '''for node in nodes.itervalues():
        if not node.children:
            continue
        assert len(node.children) == 2
    '''
    # overwrite vals to match sentence indices -
    # only leaves correspond to sentence tokens
    #leaf_idx = 0
    #for node in nodes.itervalues():
    #    if node.children:
    #        node.val = None
    #    else:
    #        node.val = leaf_idx
    #        leaf_idx += 1

    return root, max_degree

def _convert_dependencies(dependency, max_degree):
    for dep in dependency:
        dep.extend([-1] * (max_degree - len(dep)))

def _remap_tokens_and_labels(tree, sentence, fine_grained):
    # map leaf idx to word idx
    if tree.val is not None:
        tree.val = sentence[tree.val]

    [_remap_tokens_and_labels(child, sentence, fine_grained)
     for child in tree.children
     if child is not None]


def iterate_minibatches(inputs, targets, masks=None, char_inputs=None, batch_size=10, shuffle=False):
    assert len(inputs) == len(targets)
    if masks is not None:
        assert len(inputs) == len(masks)
    if char_inputs is not None:
        assert len(inputs) == len(char_inputs)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt], (None if masks is None else masks[excerpt]), \
              (None if char_inputs is None else char_inputs[excerpt])


def gen_nn_inputs(root_node, max_degree=None, only_leaves_have_vals=True,
                  with_labels=False):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    _clear_indices(root_node)
    dependencies, x, leaf_labels, indices = _get_leaf_vals(root_node, max_degree)
    #print 'leaves:', x
    #print 'leaf_labels', leaf_labels
    # tree internal nodes with the format [child, child,..., parent]; internal nodes might associalte with input or not (idx or -1); labels = local sentiment label.
    tree_dependencies, internal_x, internal_labels, internal_indices = \
        _get_tree_traversal(root_node, len(x), max_degree)
    #print 'tree internal:', internal_x
    #print 'internal_labels:', internal_labels
    assert all(v is not None for v in x)
    dependencies.extend(tree_dependencies)
    indices.extend(internal_indices)
    if not only_leaves_have_vals:
        assert all(v is not None for v in internal_x)
        x.extend(internal_x)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in dependencies)
    if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(dependencies, dtype='int32'),
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX),
                indices)
    return (np.array(x, dtype='int32'),
            #np.array(dependencies, dtype='int32'),
            np.array(tree_dependencies, dtype='int32'),
            indices)


def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node, max_degree):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if all(child is None for child in node.children):
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    dependencies = []
    vals = []
    labels = []
    indices = []
    # print 'in _get_leaf_vals, leaves:', [leaf.idx for leaf in all_leaves]
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        child_idxs = [-1] * max_degree
        dependencies.append(child_idxs + [leaf.idx])
        vals.append(leaf.val)
        labels.append(leaf.label)
        indices.append(leaf.origin_idx)
    return dependencies, vals, labels, indices


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        sys.stderr.write('Empty tree!!!!!')    
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
	layer = next_layer

    dependencies = []
    internal_vals = []
    labels = []
    indices = []
    idx = start_idx
    # print 'start index:', idx, ', num layers:', len(layers)
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            #print 'children indices for node', idx, ': ', child_idxs
            dependencies.append(child_idxs + [node.idx])
            internal_vals.append(node.val if node.val is not None else -1)
            labels.append(node.label)
            indices.append(node.origin_idx)
            idx += 1
    # dependencies internal nodes with the format [child, child,..., parent]; internal nodes might associalte with input or not (idx or -1); labels = local sentiment label.
    return dependencies, internal_vals, labels, indices


