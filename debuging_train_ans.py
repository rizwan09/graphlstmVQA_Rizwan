import json
matching_ans = ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes']

with open('vqa_data/train_all_ans.txt', "r") as train_annoattion_f:
    m_index = 0
    id = 0
    for line in train_annoattion_f:
        ans = line.strip()
        #print "ans: ", ans, " id: ", id, " mIndex: ", m_index
        if(m_index==matching_ans.__len__()):
            print "found at ended: ", id, m_index
            break
        if(ans == matching_ans[m_index]):
            m_index += 1
        else:
            m_index = 0



        id +=1


