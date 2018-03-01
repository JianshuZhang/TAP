import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def dataIterator(feature_file,label_file,dictionary,batch_size,maxlen):
    
    fp=open(feature_file,'rb') # read kaldi scp file
    features=pkl.load(fp) # load features in dict
    fp.close()

    fp2=open(label_file,'r')
    labels=fp2.readlines()
    fp2.close()

    targets={}
    # map word to int with dictionary
    for l in labels:
        tmp=l.strip().split()
        uid=tmp[0]
        w_list=[]
        for w in tmp[1:]:
            if dictionary.has_key(w):
                w_list.append(dictionary[w])
            else:
                print 'a word not in the dictionary !! sentence ',uid,'word ', w
                sys.exit()
        targets[uid]=w_list



    sentLen={}
    for uid,fea in features.iteritems():
        sentLen[uid]=len(fea)

    sentLen= sorted(sentLen.iteritems(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element


    feature_batch=[]
    label_batch=[]
    feature_total=[]
    label_total=[]
    uidList=[]

    i=0
    for uid,length in sentLen:
        fea=features[uid]
        lab=targets[uid]
        if len(lab)>maxlen:
            print 'this sentence length bigger than', maxlen, 'ignore'
        else:
            uidList.append(uid)
            if i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)

                i=0
                feature_batch=[]
                label_batch=[]
                feature_batch.append(fea)
                label_batch.append(lab)
                i=i+1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i=i+1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print 'total ',len(feature_total), 'batch data loaded'

    return zip(feature_total,label_total),uidList
