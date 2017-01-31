#!/usr/bin/python

import sys,random, re

def load_sentences_file(filename):
    with open(filename, 'r') as inf:
        sent_dict = dict()
        for line in inf:
            elems = line.strip().split('\t')
            assert len(elems) >= 3
            assert elems[0] not in sent_dict
            sent_dict[elems[0]] = elems[3]
        return sent_dict

def load_deps_or_rel_file(filename):
    with open(filename, 'r') as inf:
        inf.readline()
        dep_or_rel_dict = dict()
        for line in inf:
            elems = line.strip().split('\t')
            key = '_'.join(elems[:2])
            if key not in dep_or_rel_dict:
                dep_or_rel_dict[key] = []
            dep_or_rel_dict[key].append(elems[2:])
        return dep_or_rel_dict 

def load_entt_file(filename):
    with open(filename, 'r') as inf:
        inf.readline()
        entt_dict = dict()
        for line in inf:
            elems = line.strip().split('\t')
            assert elems[0] not in entt_dict
            entt_dict[elems[0]] = elems[1:]
        return entt_dict

def parse_stanford_dep_file(filename):
    deps_array = []
    one_sent_dep = []
    with open(filename, 'r') as inf:
        for line in inf:
            if line.strip() == '':
                deps_array.append(one_sent_dep)
                one_sent_dep = []
            else:
                elems = re.split(', |\(|\)', line.strip())
                '''
                print elems[1].split('-')
                print line.strip()
                print len(elems), elems
                print elems[1], elems[2]
                '''
                elems = [el for el in elems if el != '']
                assert len(elems) == 3 
                one_sent_dep.append(( str(int(remove_ending_quote(elems[1].split('-')[-1]))-1), str(int(remove_ending_quote(elems[2].split('-')[-1]))-1), elems[0]))
    return deps_array

def parse_universal_dep_file(filename):
    sent_dep_dict = dict()
    sentence_tokens = []
    analysis = []
    with open(filename, 'r') as inf:
        for line in inf:
            if line.strip() == '':
                key = re.sub('[\'|\.|`|:|\%|\+|<|>|\-|,|/|=|;| |0-9]+', '', ''.join(sentence_tokens).lower())
                sent_dep_dict[key] = analysis
                sentence_tokens = []
                analysis = []
            else:
                elems = line.strip().split('\t')
                sentence_tokens.append(elems[1])
                analysis.append(elems)
    return sent_dep_dict

def remove_ending_quote(string):
    while string.endswith('\''):
        string = string[:-1]
    return string

def load_docSent(filename):
    docSent_array = []
    with open(filename, 'r') as inf:
        for line in inf:
            docSent_array.append(line.strip().split('\t')[0])
    return docSent_array

def gen_pos_instances(relation_dict, entt_dict, sent_dict):
    anno_posSent_dict = dict()
    for k, rels in relation_dict.iteritems():
        sent = sent_dict[k].split(' ')
        for elem in rels:
            ett1, ett2 = elem[:2]
            ett1_info = entt_dict[ett1]
            assert '_'.join(ett1_info[:2]) == k
            fetched_entity = ' '.join(sent[int(ett1_info[2]):int(ett1_info[3])]) 
            try:
                assert (fetched_entity in ett1_info[4]) or (ett1_info[4] in fetched_entity)
            except:
                print fetched_entity, 'v.s.', ett1_info[4]
            ett2_info = entt_dict[ett2]
            assert '_'.join(ett2_info[:2]) == k
            fetched_entity = ' '.join(sent[int(ett2_info[2]):int(ett2_info[3])]) 
            try:
                assert (fetched_entity in ett2_info[4]) or (ett2_info[4] in fetched_entity)
            except:
                print fetched_entity, 'v.s.', ett2_info[4]
            if k not in anno_posSent_dict:
                anno_posSent_dict[k] = dict()
            entity_pair = ett1+'_AND_'+ett2
            try:
                assert entity_pair not in anno_posSent_dict[k] 
            except:
                print 'entity_pair', entity_pair, 'has already in the annotation dict, the relation is:', anno_posSent_dict[k][entity_pair], 'new relation:', elem[2]
            anno_posSent_dict[k][entity_pair] = (sent_dict[k], gen_index_span(ett1_info[2], ett1_info[3]), gen_index_span(ett2_info[2], ett2_info[3]), elem[2])
    return anno_posSent_dict 

def group_entity_by_sent(entt_dict):
    doc_entities = dict()
    for k, v in entt_dict.iteritems():
        doc_id = '_'.join(v[:2])
        if doc_id not in doc_entities:
            doc_entities[doc_id] = []
        doc_entities[doc_id].append(k)
    return doc_entities

def gen_neg_instances(entt_dict, anno_pos_dict, sent_dict):
    anno_neg_dict = dict()
    doc_entities = group_entity_by_sent(entt_dict)
    for k, v in doc_entities.iteritems():
        if k == '10022882_0':
            continue
        pos_examples = anno_pos_dict.get(k, None)
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                entity_pair = v[i]+'_AND_'+v[j]
                if pos_examples is None or entity_pair not in pos_examples:
                    if k not in anno_neg_dict:
                        anno_neg_dict[k] = dict()
                    assert entity_pair not in anno_neg_dict[k] 
                    anno_neg_dict[k][entity_pair] = (sent_dict[k], gen_index_span(entt_dict[v[i]][2], entt_dict[v[i]][3]), gen_index_span(entt_dict[v[j]][2], entt_dict[v[j]][3]), 'None')
                entity_pair = v[j]+'_AND_'+v[i]
                if pos_examples is None or entity_pair not in pos_examples:
                    if k not in anno_neg_dict:
                        anno_neg_dict[k] = dict()
                    assert entity_pair not in anno_neg_dict[k] 
                    anno_neg_dict[k][entity_pair] = (sent_dict[k], gen_index_span(entt_dict[v[j]][2], entt_dict[v[j]][3]), gen_index_span(entt_dict[v[i]][2], entt_dict[v[i]][3]), 'None')
    return anno_neg_dict

def gen_index_span(start, end):
    #print 'generated index:', ' '.join(map(str, range(int(start),int(end))))
    return ' '.join(map(str, range(int(start),int(end))))

def load_train_dev_split(filename):
    train_set = set()
    dev_set = set()
    with open(filename, 'r') as inf:
        inf.readline()
        for line in inf:
            elems = line.strip().split('\t')
            if elems[1] == 'trn':
                assert elems[0] not in train_set
                train_set.add(elems[0])
            else:
                assert elems[1] == 'dev'
                assert elems[0] not in dev_set 
                dev_set.add(elems[0])
    return train_set, dev_set

def gen_dep_graph(dep_dict, sent_dict, deparc_counts):
    sent_depGraph_dict = dict()
    for k, v in sent_dict.iteritems():
        assert k in dep_dict
        sent_len = len(v.split(' '))
        dep_graph = [[] for i in range(sent_len)]
        for i in range(sent_len):
            if i != 0:
                dep_graph[i].append('adjtok:prev::'+str(i-1))
            if i != sent_len-1:
                dep_graph[i].append('adjtok:next::'+str(i+1))
        dep_arcs = dep_dict[k]
        print 'key:', k, 'sent length:', sent_len, len(dep_arcs)
        #assert len(dep_arcs) == sent_len
        for arc in dep_arcs:
            assert len(arc) >= 3
            if arc[2] not in deparc_counts:
                print arc[2], 'not in dict!!'
                continue
            if arc[0] != '-1':
                dep_graph[int(arc[0])].append('depinv:'+arc[2]+'::'+arc[1])
            assert arc[1] != '-1'
            dep_graph[int(arc[1])].append('deparc:'+arc[2]+'::'+arc[0])
        sent_depGraph_dict[k] = dep_graph
    return sent_depGraph_dict

def convert_dep_dict(docSent_ids, dep_parses):
    dep_dict = dict()
    for key, val in zip(docSent_ids, dep_parses):
        assert key not in dep_dict
        dep_dict[key] = val
    return dep_dict


def generate_golden_parse(sentence_file, dep_files):
    # Hacky part to generate golden parse.
    sent_dict = load_sentences_file(sentence_file)
    sent_to_dep = dict()
    for filename in dep_files:
        sent_to_dep.update(parse_universal_dep_file(filename)) 
        print len(sent_to_dep)
    print 'number of sentence to be found:', len(sent_dict)
    found_sent_count = 0
    with open(sentence_file+'.gold_parse', 'w') as outf:
        for k, v in sent_dict.iteritems():
            sent = re.sub('[\'|\.|`|:|\%|\+|<|>|\-|,|/|=|;| |0-9]+', '', v.lower())
            try:
                assert sent in sent_to_dep
                found_sent_count += 1
                analysis = sent_to_dep[sent]
                print len(analysis)
                for line in analysis:
                    outf.write('\t'.join(k.split('_') + [str(int(line[6])-1), str(int(line[0])-1), line[7], line[1]]) + '\n' )
            except:
                pass
                #print 'sentence is not in the dict!!', k, sent
        print 'number of found sentences:', found_sent_count
        #for k in sent_to_dep:
        #    print ki

def get_deparc_count(relation_dict, dep_dict):
    deparc_counts = dict()
    for k in relation_dict:
        assert k in dep_dict
        dep_arcs = dep_dict[k]
        for arc in dep_arcs:
            arc_label = arc[2]
            if arc_label not in deparc_counts:
                deparc_counts[arc_label] = 0
            deparc_counts[arc_label] += 1
    print len(deparc_counts)
    for k, v in deparc_counts.items():
        if v < 3:
            del deparc_counts[k] 
    print len(deparc_counts)
    return deparc_counts 


if __name__ == '__main__':
    #--------------------------------------------------------------------------------------------------------------------
    # Generate gold parse from David Mcclomsky's data
    #generate_golden_parse(sys.argv[1], sys.argv[2:])
    #exit(0)
    #--------------------------------------------------------------------------------------------------------------------
    sent_dict = load_sentences_file(sys.argv[1])
    #dep_dict = load_deps_or_rel_file(sys.argv[2])
    #dep_dict.update(load_deps_or_rel_file(sys.argv[8]))
    # This is for the new stanford dep parser.
    dep_parses = parse_stanford_dep_file(sys.argv[2])
    docSent_ids = load_docSent(sys.argv[1])
    assert len(dep_parses) == len(docSent_ids)
    dep_dict = convert_dep_dict(docSent_ids, dep_parses)
    # -------------------------------------------------------------------------------------------------------------------
    entt_dict = load_entt_file(sys.argv[3])
    relation_dict = load_deps_or_rel_file(sys.argv[4])
    #assert len(sent_dict) == len(dep_dict)
    # Get the dep_arc dictionary with arcs that have more than 3 counts in pos examples.
    anno_posSent_dict = gen_pos_instances(relation_dict, entt_dict, sent_dict)
    anno_negSent_dict = gen_neg_instances(entt_dict, anno_posSent_dict, sent_dict)
    # Add dependency arc filter.
    deparc_counts = get_deparc_count(relation_dict, dep_dict)
    sent_depGraph_dict = gen_dep_graph(dep_dict, sent_dict, deparc_counts)
    train_set, dev_set = load_train_dev_split(sys.argv[5])
    dep_mode = sys.argv[6]
    sample_coef = int(sys.argv[7])
    with open('sentences.'+dep_mode+'.negSample'+sys.argv[7]+'.train', 'w') as sent_train, open('sentences.'+dep_mode+'.negSample'+sys.argv[7]+'.dev', 'w') as sent_dev, open('graph.'+dep_mode+'.negSample'+sys.argv[7]+'.train', 'w') as graph_train, open('graph.'+dep_mode+'.negSample'+sys.argv[7]+'.dev', 'w') as graph_dev:
        num_pos_examples = 0
        print 'write positive instances!!! pos dict size:', len(anno_posSent_dict)
        for k, v in anno_posSent_dict.iteritems():
            if k.split('_')[0] in train_set:
                for pair, content in v.iteritems():
                    num_pos_examples += 1
                    sent_train.write( '\t'.join(content)+'\n')
                    graph_arry = [',,,'.join(elem) for elem in sent_depGraph_dict[k]]
                    graph_train.write(' '.join(graph_arry)+'\n')
            else:
                for pair, content in v.iteritems():
                    sent_dev.write( '\t'.join(content)+'\n')
                    graph_arry = [',,,'.join(elem) for elem in sent_depGraph_dict[k]]
                    graph_dev.write(' '.join(graph_arry)+'\n')
        print 'write negative instances!!! neg dict size:', len(anno_negSent_dict)
        print 'number of positive sample:', num_pos_examples
        neg_training_instances = []
        neg_training_graph = []
        for k, v in anno_negSent_dict.iteritems():
            if k.split('_')[0] in train_set:
                for pair, content in v.iteritems():
                    neg_training_instances.append( '\t'.join(content)+'\n')
                    #sent_train.write( '\t'.join(content)+'\n')
                    graph_arry = [',,,'.join(elem) for elem in sent_depGraph_dict[k]]
                    neg_training_graph.append(' '.join(graph_arry)+'\n')
                    #graph_train.write(' '.join(graph_arry)+'\n')
            else:
                for pair, content in v.iteritems():
                    sent_dev.write( '\t'.join(content)+'\n')
                    graph_arry = [',,,'.join(elem) for elem in sent_depGraph_dict[k]]
                    graph_dev.write(' '.join(graph_arry)+'\n')
        neg_samples_idx = random.sample(range(len(neg_training_instances)), sample_coef * num_pos_examples)
        for idx in neg_samples_idx:
            sent_train.write(neg_training_instances[idx])
            graph_train.write(neg_training_graph[idx])

