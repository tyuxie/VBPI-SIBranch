
# Data Manipulation

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np

nucnames = ['A','G','C','T']

def loadData(filename,data_type):
    data = []
    id_seq = []
    for seq_record in SeqIO.parse(filename,data_type):
        id_seq.append(seq_record.id)
        data.append(list(seq_record.seq.upper()))

    return data, id_seq

def saveData(data, id_seq, filename, data_type):
    my_seq = []
    for i,seq in enumerate(data):
        my_seq.append(SeqRecord(Seq(''.join(seq)),id=str(id_seq[i]),description=''))
    
    with open(filename,"w") as output_file:
        SeqIO.write(my_seq, output_file, data_type)


def statSamp(pden, idx=False):
    cumsum = pden.cumsum()
    u = np.random.uniform()
    for i in range(4):
        if u <= cumsum[i]:
            if idx:
                return i
            else:
                return nucnames[i]