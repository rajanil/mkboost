import numpy as np
import cPickle
import generate_features
import splitdata
import boost
import pdb
import csv
import os

# set paths 
project_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
virus_family = 'picorna'
if not os.path.exists(project_path+'/cache'):
    os.makedir(project_path+'/cache')
    os.makedir('%s/cache/%s_protein' %(project_path, virus_family))
data_path = '%s/cache/%s_protein' % (project_path, virus_family)

# set problem parameters

# length of k-mer
K = 3
# max mismatch
M = 1

# generate feature space
fasta_file = ''.join([project_path,'/data/',virus_family,'virus-proteins.fasta'])
class_file = ''.join([project_path,'/data/',virus_family,'_classes.csv'])
v = generate_features.Picorna(k=K, m=M, fasta_file=fasta_file, class_file=class_file)
v.parse()

XT, Yt, kmer_dict = v.summarise()

# save data to avoid re-parsing
for m in range(M):
    out_filename = '%s/%s_virii_data_%d_%d.pkl' % (data_path, virus_family, K, m)
    f = open(out_filename,'w')
    cPickle.Pickler(f,protocol=2).dump(XT[m])
    cPickle.Pickler(f,protocol=2).dump(Yt)
    cPickle.Pickler(f,protocol=2).dump(kmer_dict)
    f.close()

# set parameters for boosting

# boosting rounds
T = 20  
# cross-validation folds
Nfold = 10
# ADT model
model = 'tree'  

# run boosting for each value of mismatch
for m in range(M):
    Xt = XT[m].astype(float)
    Yt = Yt.astype(float)
    Nt = Yt.shape[1]
    predicted_labels = np.zeros((Nt,T),dtype='int16')

    # split the data indices into `Nfold` random disjoint sets
    Fidx = splitdata.cv_multiclass_fold(Yt,Nfold)

    for fold in range(Nfold):
        # split the data and labels into train and test sets
        train_data, train_labels, test_data, test_labels \
            = splitdata.cv_split(Xt,Yt,Fidx[fold])

        # specify output file names
        filetag = model+'_%d_%d_%d' % (K,m,fold)
        output_file = '%s/output_%s.txt' % (data_path, filetag)
        handle = open(output_file,'w')
        to_write = ['round', 'kmer', 'threshold', 'train_auc', 
                    'train_acc', 'test_auc', 'test_acc', 'runtime']
        handle.write('\t'.join(to_write)+'\n')
        handle.close()

        # run Adaboost
        adt, adt_outputs, performance, predicted_labels = boost.adaboost( \
            train_data, train_labels, test_data, test_labels, T, \
            output_file=output_file, kmer_dict=kmer_dict, model=model, \
            predicted_labels=predicted_labels, test_indices=Fidx[fold])

        # save the learned model
        model_file = '%s/adt_%s.pkl' % (data_path, filetag)
        handle = open(model_file,'w')
        cPickle.dump(adt,handle)
        handle.close()

        # save algorithm performance (errors, runtime, etc)
        results_file = '%s/result_%s.pkl' % (data_path, filetag)
        handle = open(results_file,'w')
        cPickle.Pickler(handle,protocol=2).dump(adt_outputs)
        cPickle.Pickler(handle,protocol=2).dump(performance)
        handle.close()

    # output predicted labels on test data for each CV fold
    output_file = '%s/%s_virii_test_output_%d_%d.pkl' \
                % (data_path, virus_family, K, m)
    handle = open(output_file,'w')
    cPickle.Pickler(handle,protocol=2).dump(Fidx)
    cPickle.Pickler(handle,protocol=2).dump(predicted_labels)
    handle.close()
