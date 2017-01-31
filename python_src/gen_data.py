#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, random, itertools, re

# read the annotation file and generate some basic statistics for drug, gene and variant.
def quick_stat_instances(filename):
    relation_set = set()
    drug_set = set()
    gene_set = set()
    variant_set = set()
    with open(filename, 'r') as inf:
        for line in inf:
            elems = line.strip().split(' ||| ')
            variance = None
            if len(elems) == 5:
                _, pmid, drug, gene, variance = elems
            else:
                _, pmid, drug, gene = elems
            catgry, canon_name1, token, pmid, para, sent, paraSent1, start, end = drug.split('\t')
            canon_name1 = canon_name1.lower()
            drug_set.add(canon_name1)
            catgry, canon_name2, _, _, _, _, paraSent2, _, _ = gene.split('\t')
            canon_name2 = canon_name2.lower()
            gene_set.add(canon_name2)
            if variance:
                catgry, canon_name3, _, _, _, _, paraSent2, _, _ = variance.split('\t')
                canon_name3 = canon_name3.lower()
                variant_set.add(canon_name3)
                key = (canon_name1, canon_name2, canon_name3)
            else:
                key = (canon_name1, canon_name2)
            relation_set.add(key)
    print len(relation_set), len(drug_set), len(gene_set), len(variant_set)

# Generate triple candidates
def gen_candidates_triple(data_dir, out_file):
    total_candidate_num = 0
    total_drug_gene_pair = 0
    total_drug_variant_pair = 0
    outf = open(out_file, 'w')
    for sub_dir, _, _ in os.walk(data_dir):
        assert os.path.isdir(sub_dir)
        drug_gene_dict = dict()
        drug_variance_dict = dict()
        print 'in directory:', sub_dir
        for f in os.listdir(sub_dir):
            if f.endswith('gene.txt'):
                print 'processing file:', f, 'dict sizes:', len(drug_gene_dict), len(drug_variance_dict)
                update_dict(drug_gene_dict, load_drug_gene_or_variance_singleF(os.path.join(sub_dir, f), 'drug', 'gene'))
                print 'finish loading drug-gene data'
            elif f.endswith('var.txt'):
                print 'processing file:', f, 'dict sizes:', len(drug_gene_dict), len(drug_variance_dict)
                update_dict(drug_variance_dict, load_drug_gene_or_variance_singleF(os.path.join(sub_dir, f), 'drug', 'variant'))
                print 'finish loading drug-variance data'
        for k, v in drug_gene_dict.iteritems():
            total_drug_gene_pair += len(v)
        for k, v in drug_variance_dict.iteritems():
            total_drug_variant_pair += len(v)
        if len(drug_gene_dict) == 0 or len(drug_variance_dict) == 0:
            print sub_dir, 'has no drug-gene or drug-variant data!!!'
            continue
        combined_dict = combine_drug_gene_variance(drug_gene_dict, drug_variance_dict, 3)
        if len(combined_dict) == 0: 
            print sub_dir, 'has no common data!!!'
            continue
        print 'dicts sizes:', len(drug_gene_dict), len(drug_variance_dict), len(combined_dict)
        for k, v in combined_dict.iteritems():
            total_candidate_num += len(v)
            for item in v:
                outf.write(' ||| '.join([label] + list(item))+'\n')
    print 'total drug-gene pairs:', total_drug_gene_pair, 'total drug-variant pairs', total_drug_variant_pair, 'total candidates:', total_candidate_num
    outf.close()


# generate splits for cross validation. Load all drug-gene, drug-variant and drug-gene-variant tuples, and split according to drug-gene facts. 
def gen_split_train(drug_gene_file, drug_var_file, triple_file, total_fold):
    drug_gene_index_dict, drug_gene_numbers = gen_index_drug_gene_variant(drug_gene_file, 'gene')
    drug_var_index_dict, drug_var_numbers = gen_index_drug_gene_variant(drug_var_file, 'variant')
    drug_gene_var_index_dict, triple_numbers = gen_index_drug_gene_variant(triple_file, 'triple')
    print 'finish loading the dictionaries! Sizes:', len(drug_gene_index_dict), len(drug_var_index_dict), len(drug_gene_var_index_dict)
    bucket_size = drug_gene_numbers/total_fold + 1
    drug_gene_folds = [[] for i in range(total_fold)]
    drug_var_folds = [[] for i in range(total_fold)]
    triple_folds = [[] for i in range(total_fold)]
    count = 0
    for k in sorted(drug_gene_index_dict, key=lambda k: len(drug_gene_index_dict[k]), reverse=True):
    #for key in drug_gene_index_dict:
        for article in sorted(drug_gene_index_dict[k], key=lambda article: len(drug_gene_index_dict[k][article]), reverse=True):
            fd = count % total_fold
            while len(drug_gene_folds[fd]) >= bucket_size:
                count += 1
                fd = count % total_fold
            #print k, article, fd
            drug_gene_folds[fd].extend(drug_gene_index_dict[k][article])
            if tuple(k[:1]) in drug_var_index_dict:
                if article in drug_var_index_dict[tuple(k[:1])]:
                    drug_var_folds[fd].extend(drug_var_index_dict[tuple(k[:1])][article])
                    del drug_var_index_dict[tuple(k[:1])][article]
                    if len(drug_var_index_dict[tuple(k[:1])]) == 0:
                        del drug_var_index_dict[tuple(k[:1])]
            if k in drug_gene_var_index_dict:
                if article in drug_gene_var_index_dict[k]:
                    triple_folds[fd].extend(drug_gene_var_index_dict[k][article])
                    del drug_gene_var_index_dict[k][article]
                    if len(drug_gene_var_index_dict[k]) == 0:
                        del drug_gene_var_index_dict[k]
            count += 1
    if len(drug_var_index_dict) != 0:
        bucket_size = drug_var_numbers/total_fold + 1
        for k in sorted(drug_var_index_dict, key=lambda k: len(drug_var_index_dict[k]), reverse=True):
            for article in sorted(drug_var_index_dict[k], key=lambda article: len(drug_var_index_dict[k][article]), reverse=True):
                fd = count % total_fold
                while len(drug_var_folds[fd]) >= bucket_size:
                    count += 1
                    fd = count % total_fold
                #print k, article, fd
                drug_var_folds[fd].extend(drug_var_index_dict[k][article])
    if len(drug_gene_var_index_dict) != 0:
        bucket_size = triple_numbers/total_fold + 1
        for k in sorted(drug_gene_var_index_dict, key=lambda k: len(drug_gene_var_index_dict[k]), reverse=True):
            for article in sorted(drug_gene_var_index_dict[k], key=lambda article: len(drug_gene_var_index_dict[k][article]), reverse=True):
                fd = count % total_fold
                while len(triple_folds[fd]) >= bucket_size:
                    count += 1
                    fd = count % total_fold
                #print k, article, fd
                triple_folds[fd].extend(drug_gene_var_index_dict[k][article])
    print [len(item) for item in drug_gene_folds], sum([len(item) for item in drug_gene_folds])
    print [len(item) for item in drug_var_folds], sum([len(item) for item in drug_var_folds])
    print [len(item) for item in triple_folds], sum([len(item) for item in triple_folds])
    # write the data out
    for fold in range(total_fold):
        if not os.path.exists(os.path.join(os.path.dirname(drug_gene_file), 'data_folds', str(fold))):
            os.makedirs(os.path.join(os.path.dirname(drug_gene_file), 'data_folds', str(fold)))
        if not os.path.exists(os.path.join(os.path.dirname(drug_var_file), 'data_folds', str(fold))):
            os.makedirs(os.path.join(os.path.dirname(drug_var_file), 'data_folds', str(fold)))
        if not os.path.exists(os.path.join(os.path.dirname(triple_file), 'data_folds', str(fold))):
            os.makedirs(os.path.join(os.path.dirname(triple_file), 'data_folds', str(fold)))
        with open(os.path.join(os.path.dirname(drug_gene_file), 'data_folds', str(fold), os.path.basename(drug_gene_file)[:-4]), 'w') as dg_ff, \
            open(os.path.join(os.path.dirname(drug_var_file), 'data_folds', str(fold), os.path.basename(drug_var_file)[:-4]), 'w') as dv_ff, \
            open(os.path.join(os.path.dirname(triple_file), 'data_folds', str(fold), os.path.basename(triple_file)[:-4]), 'w') as tp_ff:
            for line in drug_gene_folds[fold]:
                dg_ff.write(line)
            for line in drug_var_folds[fold]:
                dv_ff.write(line)
            for line in triple_folds[fold]:
                tp_ff.write(line)


# generate unique pairs of drug-gene or/and variant, record article information and the original content.
def gen_index_drug_gene_variant(filename, mode='gene'):
    drug_gene_index_dict = dict()
    instance_num = 0
    with open(filename) as inf:
        for line in inf:
            instance_num += 1
            elems = line.strip().split(' ||| ')
            key = []
            if mode == 'gene':
                for el in elems[2:]:
                    canon_name = el.split('\t')[1]
                    key.append(canon_name)
            else:
                for el in elems[2:-1]:
                    canon_name = el.split('\t')[1]
                    key.append(canon_name)
            key = tuple(key)
            if key not in drug_gene_index_dict:
                drug_gene_index_dict[key] = dict()
            PMID = elems[1]
            if PMID not in drug_gene_index_dict[key]:
                drug_gene_index_dict[key][PMID] = []
            drug_gene_index_dict[key][PMID].append(line)
    return drug_gene_index_dict, instance_num


# for pairwise data, key is the (drug, gene/variant), value is the instances.
def load_drug_gene_or_variance_pair(filename, e1_type, e2_type):
    drug_gene_dict = {}
    with open(filename, 'r') as inf:
        for line in inf:
            elems = line.strip().split(' ||| ')
            _, pmid, drug, gene_or_variance = elems
            catgry, canon_name1, token, pmid, para, sent, paraSent1, start, end = drug.split('\t')
            canon_name1 = canon_name1.lower()
            assert catgry == e1_type
            #key = '|||'.join([canon_name1, pmid, paraSent1, start, end])
            catgry, canon_name2, _, _, _, _, paraSent2, _, _ = gene_or_variance.split('\t')
            canon_name2 = canon_name2.lower()
            assert catgry == e2_type
            key = (canon_name1, canon_name2)
            value = (pmid, drug, gene_or_variance)
            #assert key not in drug_gene_dict
            if key not in drug_gene_dict:
                drug_gene_dict[key] = []
            drug_gene_dict[key].append(value)
    return drug_gene_dict

# for triple
# store the information in a dict for a quick comparison. 
# dict key: drug_canon_name+pmid+paraSent+offset
# value: tuple: ((PMID, gene_canon_name, paraSent_drug, paraSent_gene), concatenation of the two entity contents) 
def load_drug_gene_or_variance_singleF(filename, e1_type, e2_type):
    drug_gene_dict = {}
    with open(filename, 'r') as inf:
        for line in inf:
            elems = line.strip().split(' ||| ')
            _, pmid, drug, gene_or_variance = elems
            catgry, canon_name1, token, pmid, para, sent, paraSent1, start, end = drug.split('\t')
            canon_name1 = canon_name1.lower()
            assert catgry == e1_type
            key = '|||'.join([canon_name1, pmid, paraSent1, start, end])
            catgry, canon_name2, _, _, _, _, paraSent2, _, _ = gene_or_variance.split('\t')
            canon_name2 = canon_name2.lower()
            assert catgry == e2_type
            value = ((pmid, canon_name1, canon_name2, paraSent1, paraSent2), (drug, gene_or_variance))
            #assert key not in drug_gene_dict
            if key not in drug_gene_dict:
                drug_gene_dict[key] = []
            drug_gene_dict[key].append(value)
    return drug_gene_dict


# match the drug-key, decide whether the sentances are within the window_size, if yes, claim instances.
# output: a dictionary with key (drug, gene, variance), value (PMID, drug_content, gene_content, variance_content) 
def combine_drug_gene_variance(drug_gene, drug_variance, window_size):
    combined_dict = dict()
    for d, g in drug_gene.iteritems():
        if d in drug_variance:
            v = drug_variance[d]
            update_dict(combined_dict, gen_candidates(d, g, v, window_size))
    return combined_dict

# output: a dictionary with key (drug, gene, variance), value (PMID, drug_content, gene_content, variance_content) 
def gen_candidates(drug_key, gene_list, variance_list, window_size):
    candidate_dict = dict()
    for g in gene_list:
        for v in variance_list:
            idx1, idx2 = g[0][-2:]
            idx3 = v[0][-1]
            idx_list = [idx1, idx2, idx3]
            if max(map(int,idx_list)) - min(map(int,idx_list)) < window_size:
                assert g[0][1] == v[0][1]
                key = (g[0][1], g[0][2], v[0][2])
                value = (g[0][0], g[1][0], g[1][1], v[1][1])
                if key not in candidate_dict:
                    candidate_dict[key] = []
                candidate_dict[key].append(value)
    return candidate_dict

# This is different from dict.update(additional_dict), in the sense that we want to extend the value list
def update_dict(orig_dict, additional_dict):
    for k, v in additional_dict.iteritems():
        if k not in orig_dict:
            orig_dict[k] = []
        orig_dict[k].extend(v)

# return a dictionary with key (drug, gene, variance), value the relation-string
def load_civic_knowledge_base(filename, mode='triple'):
    kb_dict = dict()
    splitter = ',|\+'
    with open(filename) as inf:
        line_num = 0
        for line in inf:
            elems = line.lower().split('\t')
            if line_num == 0:
                # proces the header
                header_dict = dict((k,v) for k, v in zip(elems, range(len(elems))))
            else:
                label = elems[header_dict['clinical_significance']]
                if label != 'sensitivity' and label != 'resistance or non-response':
                    continue
                # get the key information
                if mode == 'triple':
                    for drug, gene, variant in itertools.product(re.split(splitter, elems[header_dict['drugs']].lower()), re.split(splitter, elems[header_dict['gene']]), re.split(splitter, elems[header_dict['variant']])):
                        key = (drug.strip(), gene.strip(), variant.strip())
                        try:
                            assert (key not in kb_dict or kb_dict[key] == label)
                        except:
                            sys.stderr.write(str(key) + '\t' + str(kb_dict[key]) + '\t' + label + '\n')
                            continue
                        kb_dict[key] = label
                elif mode == 'gene':
                    for drug, gene in itertools.product(re.split(splitter, elems[header_dict['drugs']].lower()), re.split(splitter, elems[header_dict['gene']])):
                        key = (drug.strip(), gene.strip())
                        try:
                            assert (key not in kb_dict or kb_dict[key] == label)
                        except:
                            sys.stderr.write(str(key) + '\t' + str(kb_dict[key]) + '\t' + label + '\n')
                            continue
                        kb_dict[key] = label
                elif mode == 'variant':
                    for drug, variant in itertools.product(re.split(splitter, elems[header_dict['drugs']].lower()), re.split(splitter, elems[header_dict['variant']])):
                        key = (drug.strip(), variant.strip())
                        try:
                            assert (key not in kb_dict or kb_dict[key] == label)
                        except:
                            sys.stderr.write(str(key) + '\t' + str(kb_dict[key]) + '\t' + label + '\n')
                            #del kb_dict[key]
                            continue
                        kb_dict[key] = label
            line_num += 1
    print 'kb size:', len(kb_dict)
    return kb_dict


# return a dictionary with key (drug, gene, variance), value the relation-string
def load_dgkd_knowledge_base(filename, mode='triple'):
    kb_dict = dict()
    splitter = ',|\+'
    with open(filename) as inf:
        line_num = 0
        for line in inf:
            elems = line.lower().split('\t')
            if line_num == 0:
                # proces the header
                header_dict = dict((k.strip(),v) for k, v in zip(elems, range(len(elems))))
            else:
                for i in range(1, 9):
                    label = elems[header_dict['association_'+str(i)]]
                    if label.strip() == '': 
                        continue
                    # get the key information
                    if mode == 'triple':
                        for drug, gene, variant in itertools.product(re.split(splitter, elems[header_dict['therapeutic context_'+str(i)]].lower()), re.split(splitter, elems[header_dict['gene']]), re.split(splitter, elems[header_dict['variant']])):
                            key = (drug.strip(), gene.strip(), variant.strip())
                            try:
                                assert (key not in kb_dict or kb_dict[key] == label)
                            except:
                                sys.stderr.write(str(key) + '\t' + str(kb_dict[key]) + '\t' + label + '\n')
                                continue
                            kb_dict[key] = label
                    if mode == 'gene':
                        for drug, gene in itertools.product(re.split(splitter, elems[header_dict['therapeutic context_'+str(i)]].lower()), re.split(splitter, elems[header_dict['gene']])):
                            key = (drug.strip(), gene.strip())
                            try:
                                assert (key not in kb_dict or kb_dict[key] == label)
                            except:
                                sys.stderr.write(str(key) + '\t' + str(kb_dict[key]) + '\t' + label + '\n')
                                continue
                            kb_dict[key] = label
                    if mode == 'variant':
                        for drug, variant in itertools.product(re.split(splitter, elems[header_dict['therapeutic context_'+str(i)]].lower()), re.split(splitter, elems[header_dict['variant']])):
                            key = (drug.strip(), variant.strip())
                            try:
                                assert (key not in kb_dict or kb_dict[key] == label)
                            except:
                                sys.stderr.write(str(key) + '\t' + str(kb_dict[key]) + '\t' + label + '\n')
                                continue
                            kb_dict[key] = label
            line_num += 1
    print 'kb size:', len(kb_dict)
    return kb_dict


# given knowledge base dictionary and the database dictionary, match the entires. If matched, positive examples.
def gen_pos_instances(knowledge_base, database):
    pos_samples = []
    for kb_triple, label in knowledge_base.iteritems():
        if kb_triple in database:
            for item in database[kb_triple]:
                pos_samples.append(' ||| '.join([label] + list(item)))
    return pos_samples


def gen_neg_instances(knowledge_base, database, num_instances):
    neg_samples = []
    for k, v in database.iteritems():
        if k not in knowledge_base:
            for item in v:
                neg_samples.append(' ||| '.join(['None'] + list(item)))
    if len(neg_samples) < num_instances * 1.1:
        return neg_samples
    return random.sample(neg_samples, int(num_instances*1.1))
    

def gen_instances_pairwise(data_dir, kb_dict, pos_file, neg_file, target_name):
    posf = open(pos_file, 'w')
    negf = open(neg_file, 'w')
    total_candidate_num = 0
    type_name = 'gene'
    if target_name == 'variant':
        type_name = 'var'
    for sub_dir, _, _ in os.walk(data_dir):
        assert os.path.isdir(sub_dir)
        drug_gene_or_variant_dict = dict()
        print 'in directory:', sub_dir
        for f in os.listdir(sub_dir):
            if f.endswith(type_name + '.txt'):
                update_dict(drug_gene_or_variant_dict, load_drug_gene_or_variance_pair(os.path.join(sub_dir, f), 'drug', target_name))
                print 'processing file:', f, 'dict size:', len(drug_gene_or_variant_dict)
        if len(drug_gene_or_variant_dict) == 0: 
            print 'no drug-gene or drug-variant data!!!'
            continue
        max_len = 0
        for k, v in drug_gene_or_variant_dict.iteritems():
            #print k, len(v)
            total_candidate_num += len(v)
            if len(v) > max_len:
                max_len = len(v)
        print 'max number of instances:', max_len
        pos_samples = gen_pos_instances(kb_dict, drug_gene_or_variant_dict)
        neg_samples = gen_neg_instances(kb_dict, drug_gene_or_variant_dict, len(pos_samples))
        sys.stderr.write('num samples: ' + str(len(pos_samples)) + '\n' )#+ str(len(neg_samples))+'\n')
        if len(pos_samples) != 0:
            #sys.stderr.write(pos_samples[0]+'\n')
            #sys.stderr.write(neg_samples[0]+'\n')
            for pos in pos_samples:
                posf.write(pos + '\n')
            for neg in neg_samples:
                negf.write(neg + '\n')
        else:
            sys.stderr.write('print data dict:\n')
            for k in drug_gene_or_variant_dict:
                sys.stderr.write(str(k)+'\n')
    print 'total candidates:', total_candidate_num
    posf.close()
    negf.close()


def gen_instances_triple(data_dir, kb_dict, pos_file, neg_file):
    posf = open(pos_file, 'w')
    negf = open(neg_file, 'w')
    total_candidate_num = 0
    total_drug_gene_pair = 0
    total_drug_variant_pair = 0
    for k in kb_dict:
        sys.stderr.write(str(k)+'\n')
    for sub_dir, _, _ in os.walk(data_dir):
        assert os.path.isdir(sub_dir)
        drug_gene_dict = dict()
        drug_variance_dict = dict()
        print 'in directory:', sub_dir
        for f in os.listdir(sub_dir):
            if f.endswith('gene.txt'):
                print 'processing file:', f, 'dict sizes:', len(drug_gene_dict), len(drug_variance_dict)
                update_dict(drug_gene_dict, load_drug_gene_or_variance_singleF(os.path.join(sub_dir, f), 'drug', 'gene'))
                print 'finish loading drug-gene data'
            elif f.endswith('var.txt'):
                print 'processing file:', f, 'dict sizes:', len(drug_gene_dict), len(drug_variance_dict)
                update_dict(drug_variance_dict, load_drug_gene_or_variance_singleF(os.path.join(sub_dir, f), 'drug', 'variant'))
                print 'finish loading drug-variance data'
        temp_dg_pairs = 0
        temp_dv_pairs = 0
        temp_cd_num = 0
        for k, v in drug_gene_dict.iteritems():
            total_drug_gene_pair += len(v)
            temp_dg_pairs += len(v)
        for k, v in drug_variance_dict.iteritems():
            total_drug_variant_pair += len(v)
            temp_dv_pairs += len(v)
        if len(drug_gene_dict) == 0 or len(drug_variance_dict) == 0:
            print 'no drug-gene or drug-variant data!!!'
            continue
        combined_dict = combine_drug_gene_variance(drug_gene_dict, drug_variance_dict, 3)
        if len(combined_dict) == 0: 
            print 'no common data!!!'
            continue
        print 'dicts sizes:', len(drug_gene_dict), len(drug_variance_dict), len(combined_dict)
        max_len = 0
        for k, v in combined_dict.iteritems():
            #print k, len(v)
            total_candidate_num += len(v)
            temp_cd_num += len(v)
            if len(v) > max_len:
                max_len = len(v)
        print 'max number of instances:', max_len
        pos_samples = gen_pos_instances(kb_dict, combined_dict)
        neg_samples = gen_neg_instances(kb_dict, combined_dict, len(pos_samples))
        sys.stderr.write('num samples: ' + str(len(pos_samples)) + '\n' )#+ str(len(neg_samples))+'\n')
        if len(pos_samples) != 0:
            #sys.stderr.write(pos_samples[0]+'\n')
            #sys.stderr.write(neg_samples[0]+'\n')
            for pos in pos_samples:
                posf.write(pos + '\n')
            for neg in neg_samples:
                negf.write(neg + '\n')
        else:
            sys.stderr.write('print combined dict:\n')
            for k in combined_dict:
                sys.stderr.write(str(k)+'\n')
    print 'total drug-gene pairs:', total_drug_gene_pair, 'total drug-variant pairs', total_drug_variant_pair, 'total candidates:', total_candidate_num
    posf.close()
    negf.close()


if __name__ == '__main__':
    kb_dict = load_civic_knowledge_base(sys.argv[1], sys.argv[3])
    kb_dict.update(load_dgkd_knowledge_base(sys.argv[2], sys.argv[3]))
    print 'kb size:', len(kb_dict)
    exit(0)
    eval(sys.argv[1])(*sys.argv[2:])    
'''    data_dir = sys.argv[1]
    civic_kb_file = sys.argv[2]
    dgkd_kb_file = sys.argv[3]
    pos_file = sys.argv[4]
    neg_file = sys.argv[5]
    mode = sys.argv[6]
    #kb_dict = load_civic_knowledge_base(civic_kb_file, mode)
    kb_dict = load_dgkd_knowledge_base(dgkd_kb_file, mode)
    #kb_dict.update(load_dgkd_knowledge_base(dgkd_kb_file, mode))
    sys.stderr.write('print kb dict:\n')
    #gen_instances_triple(data_dir, kb_dict, pos_file, neg_file)
    gen_instances_pairwise(data_dir, kb_dict, pos_file, neg_file, mode)
'''
