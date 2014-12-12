import urllib
import json
import time
import mismatch
import csv
import numpy as np
import cPickle
import os
import pdb

class Protein():
    """
    Class describing a protein in terms of its amino acid sequence

    Arguments
        name : string
            name of the protein

    """
    def __init__(self,name):

        print "\tinitialised %s"%name
        self.name = name
        self.lines = []
        self.label = None
    
    def add_line(self,line):
        """
        this adds a line of symbols to the (temporary) `lines` variable
        
        Arguments
            line : string
                a line of protein symbols from a fasta file

        """
        self.lines.append(line)
        
    def finish(self,m,beta):
        """
        this finishes off the parsing of a single protein from a fasta file.
        It also generates the feature set - the count of each possible kmer in
        the protein that are within m mismatches.

        Arguments
            m : int
                number of mismatches allowed
            beta : list
                all possible kmers

        """
        print "\tfinishing %s"%self.name
        self.data = "".join(self.lines)
        print "\t\tgenerating features"
        self.feature = mismatch.gen_features(self.data,m,beta)
        
    def __str__(self):
        return self.name + "\n" + self.data

class Virus():
    """
    class describing a virus as a collection of proteins

    Arguments
        name : string
            name of the virus
        virus_id : string
            unique id of the virus
        m : int
            number of allowed mismatches
        beta : list
            all possible kmers

    .. note::
        The arguments `m` and `beta` are for generating features. See 
        :func:`protein.finish` for more info.

    """

    def __init__(self,name,virus_id,m,beta):

        print "initialised %s with id %s"%(name,virus_id)
        self.name = name
        self.id = virus_id
        self.proteins = []
        self.label = None
        self.m = m
        self.beta = beta
        
    def add_line(self,line):
        """
        adds a line from a fasta file to the virus definition. This either
        starts a new protein or adds a line to the current protein.

        Arguments
            line : str

        """
        if ">" in line:
            if len(self.proteins):
                self.proteins[-1].finish(self.m,self.beta)
            self.proteins.append(Protein(line))
        else:
            self.proteins[-1].add_line(line)
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self,i):
        return self.proteins[i]
    
    def __str__(self):
        return self.name

class Picorna():
    """
    class describing a set of picorna viruses

    Arguments
        k : int
            length of kmers to consider
        m : int
            largest number of mismatches
        fasta_file : str
            full path of file containing raw sequence data
        class_file : str
            full path of file containing virus host labels

    """

    def __init__(self,k,m,fasta_file,class_file):

        self.k = k
        self.m = m
        self.fasta_file = fasta_file
        self.class_file = class_file

        # form all kmers observed in data
        cmd = [
            "grep -v NC_ %s" % self.fasta_file,
            "grep -v '>'",
            "tr '\n' '*'"
        ]
        x = os.popen(' | '.join(cmd)).next()
        self.beta = mismatch.form_all_kmers_in_string(self.k,x)

        self.viruses = []
        # a dictionary of label numbers to labels
        # replace elements for rhabdo viruses
        #1:"plant",
        #2:"animal"
        self.label_dict = {
            1:"invertebrate",
            2:"plant",
            3:"vertebrate"
        }
    
    def parse(self,max_v=None):
        """
        This method parses a fasta file, populating the objects as it goes.
        
        Kwargs
            max_v : int
                maximum number of viruses you want - used for debugging

        """
        f = open(self.fasta_file,'r').readlines()
        f = [fi.strip() for fi in f]
        for line in f:
            if ("NC_" in line or "virus" in line) and ">" not in line:
                full_name = line.split(",")[0]
                name_elements = full_name.split(' ')
                virus_name = ' '.join(name_elements[1:])
                virus_id = name_elements[0]
                self.finish_last_protein()
                if max_v:
                    if len(self.viruses) > max_v:
                        break
                self.viruses.append(Virus(virus_name,virus_id,self.m,self.beta))
            else:
                self.viruses[-1].add_line(line)
        self.finish_last_protein()
        self.assign_classes()
    
    def finish_last_protein(self):
        """
        this is called at the very end of the parsing to finish off the last
        protein.
        """
        if len(self.viruses):
            if len(self.viruses[-1].proteins):
                self.viruses[-1].proteins[-1].finish(self.m,self.beta)
    
    def assign_classes(self):
        """
        This class reads the class_file which contains the ids, names and class
        labels, and associates the appropriate label with each virus and
        protein stored in the Picorna object.
        """
        
        for row in csv.reader(open(self.class_file,'r'), delimiter=','):
            try:
                name, cls = row
            except ValueError:
                print row
                raise
            name_elements = name.split(' ')
            virus_id = name_elements[0]
            try:
                virus = self.get_virus_by_id(virus_id)
            except LookupError:
                print "can't find virus %s with id %s"%(name, virus_id)
            virus.label = self.label_dict[int(cls)]
            for protein in virus.proteins:
                protein.label = self.label_dict[int(cls)]
    
    def __len__(self):
        return len(self.viruses)
    
    def __getitem__(self,i):
        return self.viruses[i]
    
    def get_virus_by_id(self,id):
        """Returns the virus object corresponding to a given id.

        Arguments
            id : string
                id (name) of a given virus

        Raises
            `LookupError`
    
        """
        for v in self.viruses:
            if v.id == id:
                return v
        raise LookupError(id)
    
    def summarise(self):
        """
        This method collects together all the feature and class label 
        information in the Picorna object and creates a data matrix and a 
        class matrix
        
        Returns
            X : :math:`D \\times N` array
                where D = number of kmers, N = number of proteins and the array 
                elements are the kmer counts within the mismatch value
            Y : :math:`L \\times N` array
                where K = number of classes and :math:`Y_{ij} = 1` if the 
                :math:`j^{th}` protein belongs to the :math:`i^{th}` class, 
                otherwise :math:`Y_{ij} = -1`
            kmer_dict : dict
                a mapping from the row index of `X` to each of the D kmers

        """
        X = []
        for mi in range(self.m):
            feature_list = []
            for virus in self:
                # virus gets kmer counts in all its proteins
                feature_list.append(np.array([protein.feature[:,mi] for protein in virus]).sum(0))
            X.append(np.array(feature_list).T)
        
        Y = np.empty((len(self.label_dict), X[0].shape[1]))
        for i in range(Y.shape[0]):
            for j, virus in enumerate(self):
                if virus.label == self.label_dict[i+1]:
                    Y[i,j] = 1
                else:
                    Y[i,j] = -1
        kmer_dict = dict(zip(range(len(self.beta)),self.beta))
        return X, Y, kmer_dict


if __name__=="__main__":
    import csv
    # specify K,M values and virus family
    # M is the largest mismatch allowed
    K = 15
    M = 2
    project_path = '/proj/ar2384/picorna/'
    virus_family = 'picorna'
    fasta_file = ''.join([project_path,'data/',virus_family,'virus-proteins.fasta'])
    class_file = ''.join([project_path,'data/',virus_family,'_classes.csv'])
    v = Picorna(k=K, m=M, fasta_file=fasta_file, class_file=class_file)
    v.parse()

    Xt, Yt, D = v.summarise()

    # save data to avoid re-parsing
    for m in range(M):
        out_filename = '%scache/%s_protein/%s_virii_data_%d_%d.pkl' % (project_path, virus_family, virus_family, K, m)
        f = open(out_filename,'w')
        cPickle.Pickler(f,protocol=2).dump(Xt[m])
        cPickle.Pickler(f,protocol=2).dump(Yt)
        cPickle.Pickler(f,protocol=2).dump(D)
        f.close()
