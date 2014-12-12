import numpy as np
import cPickle
import matplotlib.pyplot as plot
from matplotlib.colors import colorConverter as convert
import pdb
import sys

def compile_hit_matrix(sequences, kmers, m):
    """this function compiles a matrix representation of where
    along a collection of sequences, a list of kmers are found,
    up to a given mismatch

    Arguments
        sequences : list
            List of tuples containing virus protein sequence and
            virus host class
        kmers : list
            List of kmers selected by Adaboost
        m : int
            Allowed mismatch

    Returns
        hitmatrix : int array 
            `N_seq` x `col_size` x `N_kmer` array where `N_seq` is total 
            number of viruses, `col_size` is the resolution along 
            the virus sequence and `N_kmer` is the number of kmers.

    .. note::
        In generating this hit matrix, the virus sequences are
        NOT aligned using any alignment tool. Instead, the
        sequences are simply normalized to unit length. Thus,
        location along sequence actually indicates fraction of 
        whole sequence length.

    """

    col_size = 500
    N_sequences = len(sequences)
    N_kmers = len(kmers)
    hit_matrix = np.zeros((N_sequences,col_size+1,N_kmers),dtype='int')
    kmer_length = len(kmers[0])

    for index, seq in enumerate(sequences):
        # first column stores the virus class
        hit_matrix[index,0,:] = seq[1]
        sequence = seq[0]
        sequence_length = len(sequence)
        for c in xrange(sequence_length-kmer_length+1):
            for kidx, kmer in enumerate(kmers):
                mismatch = (np.array(list(sequence[c:c+kmer_length]))!=np.array(list(kmer))).sum()
                if mismatch<=m:
                    left_col = int(c * float(col_size) / (sequence_length-kmer_length+1)) + 1
                    right_col = min([left_col+2,sequence_length-kmer_length+1])
                    # 1 indicates a match for the kmer was found
                    # at that location
                    hit_matrix[index,left_col:right_col,kidx] = 1

    return hit_matrix

def plot_hit_matrix(hit_matrix, k, m, kmers):
    """this function visualizes the kmers along protein sequences,
    represented as a matrix, using `imshow`.

    Arguments
        hit_matrix : int array
            `N_seq` x `col_size` x `N_kmer` array where `N_seq` is total 
            number of viruses, `col_size` is the resolution along 
            the virus sequence and `N_kmer` is the number of kmers.
        k : int
            size of k-mers
        m : int
            allowed mismatch
        kmers : list
            k-mers selected by Adaboost

    Returns
        figure : matplotlib figure object

    """

    # background color, text color and fontsizes
    white = np.array([255,255,255]).reshape(1,3)/255.
    text_color = 'k'
    bg_color = 'w'
    axis_label_fontsize = 7
    axis_tick_fontsize = 6
    title_fontsize = 8
    legend_fontsize = 4
    kmer_colors = ['red','green','blue','purple','magenta','orange','cyan','black','hotpink']

    class_labels = np.unique(hit_matrix[:,0,0]).astype(int)
    num_classes = class_labels.size
    (num_proteins,num_cols,ig) = hit_matrix.shape
    num_cols = num_cols-1

    # show a max of 9 kmers on the visualization
    V = min([9,len(kmers)])
    data = np.zeros((num_proteins,num_cols,3),dtype=float)

    # for each k-mer, generate a matrix where occurence
    # of the k-mer is indicated by the rgb of the color
    # assigned to that k-mer
    for i in range(V):
        data += hit_matrix[:,1:,i:i+1] * np.array(list(convert.to_rgb(kmer_colors[i]))) 

    # at every location a kmer is not found,
    # assign rgb of background color
    idx = hit_matrix.shape[0] 
    data = data + (1-(data.sum(2)>0)).reshape(idx,num_cols,1) * white

    # set figure size and resolution
    DPI = 500
    fig_resolution = (1706, 1280)
    fig_size = tuple([res/float(DPI) for res in fig_resolution])
    figure = plot.figure(figsize=fig_size, facecolor=bg_color, edgecolor=bg_color)
    subplot = figure.add_subplot(111)
    subplot.set_position([0.05,0.05,0.78,0.85])
    subplot.imshow(data,aspect='auto',interpolation='nearest')

    # set X-axis and Y-axis labels, tick sizes and tick labels
    for label in class_labels[:-1]:
        y_coord = (hit_matrix[:,0,0]==label).nonzero()[0].max() + 0.5
        subplot.plot([0,hit_matrix.shape[1]-1], [y_coord, y_coord], '-', color='gray', linewidth=0.1)

    subplot.axis([0, hit_matrix.shape[1]-1, 0, hit_matrix.shape[0]-1])
    subplot.set_xticks([0, hit_matrix.shape[1]/2, hit_matrix.shape[1]-1])
    subplot.set_xticklabels(('0','Relative Location','1'), color=text_color, \
        verticalalignment='center', fontsize=axis_tick_fontsize)
    for line in subplot.get_xticklines():
        line.set_markersize(0)
    y_labels = ('Invertebrate','Plant','Vertebrate')
    y_label_loc = []
    for c in class_labels:
        y_label_loc.append(int(np.mean((hit_matrix[:,0,0]==c).nonzero()[0])))
    subplot.set_yticks(y_label_loc)
    subplot.set_yticklabels(y_labels, rotation=90, color=text_color, \
        horizontalalignment='center', fontsize=axis_tick_fontsize)
    for line in subplot.get_yticklines():
        line.set_markersize(0)

    # set figure title
    figure.suptitle('k = %d, m = %d' % (k,m), x=0.95, y=0.95, color=text_color, \
        fontsize=title_fontsize, verticalalignment='center', horizontalalignment='right')

    # set a figtext bbox (outside the visualization subfigure) for legend
    kmer_locs = np.linspace(0.5+V/2*0.04,0.5-V/2*0.04,V)
    for kidx in range(V):
        kmer = kmers[kidx]
        plot.figtext(0.98, kmer_locs[kidx], kmer, fontsize=legend_fontsize, \
            color=kmer_colors[kidx], horizontalalignment='right', verticalalignment='center')

    return figure

if __name__=="__main__":

    # set project path and parameters
    project_path = '/proj/ar2384/picorna'
    virus_family = 'picorna'
    data_path = '%s/cache/%s' % (project_path, virus_family)
    
    # values of k, m, T and fold are read from std input
    # T = max number of boosting rounds that will be parsed
    # fold = CV fold
    (k, m, T, fold) = map(int,sys.argv[1:5])
    
    # load virus classes
    classes = dict()
    c = open('%s/data/%s_classes.csv' %(project_path,virus_family),'r')
    for line in c:
        row = line.strip().split(',')
        virus_name = ' '.join(row[0].split()[1:])
        classes[row[0].split()[0]] = [virus_name,int(row[1])]
    c.close()

    # load kmers selected by boosting
    # if a kmer is selected multiple times (with different thresholds), 
    # it is only added once to the list.
    kmers = []
    f = open('%s/adt_%s_%d_%d_%d.pkl' % (data_path, virus_family, model, k, m, fold),'r')
    adt = cPickle.load(f)
    f.close()
    [kmers.append(adt[t][0].name) for t in range(T) if adt[t][0].name not in kmers] 

    # load virus protein sequences
    p = open('%s/data/%svirus-proteins.fasta' %(project_path, virus_family),'r')
    sequences = []
    sequence = 'A'
    label = 0
    viruses = []
    for line in p:
        # indicates start of a new virus
        if 'NC_' in line or ('virus' in line and ">" not in line):
            # add previous protein to list of sequences
            sequences.append([sequence,label])
            row = line.strip().split(',')
            virus_name = ' '.join(row[0].split()[1:])
            virus_id = row[0].split()[0]
            viruses.append(virus_id)
            try:
                label = classes[virus_id][1]
            except KeyError:
                pdb.set_trace()
            sequence = ''
        # indicates start of a new protein
        elif '>' in line:
            continue
        # continue with previous protein
        else:
            sequence += line.strip()
    p.close()
    # pop out the first dummy sequence
    sequences.pop(0)

    # matrix representation of position along a sequence
    # where a specific k-mer is found, up to m mismatches
    hit_matrix = compile_hit_matrix(sequences,kmers,m)

    # save compiled data
    f = open('%s/%s_hitmatrix_%d_%d_%d.pkl' % (data_path, virus_family, k, m, fold),'w')
    cPickle.Pickler(f,protocol=2).dump(hit_matrix)
    cPickle.Pickler(f,protocol=2).dump(viruses)
    cPickle.Pickler(f,protocol=2).dump(classes)
    f.close()

    # group viruses with similar hosts together
    sort_indices = hit_matrix[:,0,0].argsort()
    sort_virus_id = [viruses[i] for i in sort_indices]
    sort_viruses = [classes[v][0] for v in sort_virus_id]

    # plot and save the visualization
    figure = plot_hit_matrix(hit_matrix[sort_indices,:,:], k, m, kmers)
    filename = '%s/fig/%s_%s_kmer_visualization_%d_%d_%d.eps' % (project_name, virus_family, k, m, fold)
    figure.savefig(fname,dpi=(500),format='eps')
