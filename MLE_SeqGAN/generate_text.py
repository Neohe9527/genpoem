#coding:utf-8
import sys
def read_dict(_dict):
    word_dict={}
    for line in _dict:
        line= line.strip().decode('utf-8').split('\t')
        word= line[0].strip()
        _id = line[1].strip()
        #print word, _id
        word_dict[int(_id)]= word
    return word_dict

def read_id(fr, _dict):
    word_dict= read_dict(_dict)
    #print word_dict
    for line in fr:
        line= line.strip().split()
        poem=[]
        seg=[]
        index=0
        for _id in line:
            index+=1
            if int(_id)==0:
                seg.append("EOS")
            else:   
                word= word_dict[int(_id)]
                seg.append(word)
            if index%7==0:
                poem.append("".join(seg))
                seg=[]
        print "\t".join(poem)

def main():
    word_dict = open(sys.argv[1], 'r')
    text= open(sys.argv[2], 'r')
    read_id(text, word_dict)

if __name__=="__main__":
    main()
