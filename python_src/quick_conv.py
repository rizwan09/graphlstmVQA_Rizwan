#!/usr/bin/python

import sys
import codecs as cs
import random

# Convert data fron conll format to the "relation format"
def get_data(fn) :
    with cs.open(fn, 'r', encoding='utf-8') as f:
        sentences = f.read().strip().split('\n\n')
        for sent in sentences:
            a = [elem.split('\t') for elem in sent.split('\n')]
            words, labels = zip(*a)
            tag, entity_ids = pos_or_neg(labels)
            print (''.join(words) + '\t' + '\t'.join([' '.join(map(str,elem)) for elem in entity_ids]) + '\t' + str(tag) )
            

# give a judge on whether an instance should be a pos or neg instance, based on its label.
# Synthetic data: positive if there are at least 2 entities.
# Return: pos/neg, 1st entity idex; 2nd entity idx
def pos_or_neg(labels):
    has_entity = False
    has_gap = False
    entity_ids = []
    length = len(labels)
    for i,lb in enumerate(labels):
        if lb.startswith('B'):
            if not has_entity:
                if i+1 == length:
                    break
                has_entity = True
                for j in range(i+1, length):
                    if not labels[j].startswith('I'):
                        break
                entity_ids.append(range(i,j))
            elif has_gap:
                for j in range(i+1, length):
                    if not labels[j].startswith('I'):
                        break
                entity_ids.append(range(i,j))
                break
        if lb.startswith('O') and has_entity:
            has_gap = True
    if len(entity_ids) >= 2:
        return True, entity_ids
    else:
        while len(entity_ids) < 2:
            start = random.randint(0, length-1)
            span = random.randint(1,4)
            entity_ids.append(range(start, start+span))
        return False, entity_ids


if __name__ == '__main__':
    get_data(sys.argv[1])
