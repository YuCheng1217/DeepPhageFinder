import os, sys
from Bio.Seq import Seq
import numpy as np
import optparse
from multiprocessing import Pool, Manager
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

prog_base = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-i", "--inputfile", action = "store", type = "string", dest = "inputfile", help = "inputfile")
parser.add_option("-l", "--contigLength", action = "store", type = int, dest = "contigLength",default = 3000, help = "contigLength")
parser.add_option("-p", "--contigType", action = "store", type = "string", dest = "contigType", help = "contigType, phage or host")
parser.add_option("-t", "--threads", action = "store", type = "int", dest = "threads", default = 1, help = "number of threads")
parser.add_option("-o", "--out", action = "store", type = "string", dest = "output_dir", default='./encode/', help = "output directory")
(options, args) = parser.parse_args()
if (options.inputfile is None or options.contigType is None ) :
        sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
        parser.print_help()
        sys.exit(0)

contigType = options.contigType
contigLength = options.contigLength
threads = options.threads
outDir = options.output_dir

inputfile = options.inputfile
if not os.path.exists(outDir):
    os.makedirs(outDir)


def clean_save(header,seq):
# split sequence when character is not A, T, C or G, sequence which length is less 3000 will be discarded
    i = 0
    j = 0
    start = 0
    end = 0
    while i < len(seq):
        if seq[i] not in primary_letter:
            if (i - start)>=3000:
                end = i
                data_list.append((header + '_' + str(j),seq[start:end]))
                j = j + 1
            start = i + 1
        i = i + 1
    if (i - start)>=3000:
        end = i
        data_list.append((header + '_' + str(j),seq[start:end]))

def encode(seq) :
# one-hot encoding
    values = list(seq)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(dtype='int32', sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def multi_encode(data) :
# encode sequence,split and save encoding results
    header = data[0]
    seq = data[1]
    seq_code=encode(seq)
    seqR = Seq(seq).reverse_complement()
    seq_codeR = encode(seqR)
    split_save(header,seq_code)
    split_save_R(header,seq_codeR)
    
def split_save(header,code) :
    k = 0
    len_assembly = len(code)//contigLength
    for i in range(len_assembly):
        segment_dic[header + '_' + str(i)] = code[k:k+contigLength]
        k = k + contigLength
def split_save_R(header,code) :
    k = 0
    len_assembly = len(code)//contigLength
    for i in range(len_assembly):
        segment_dic_R[header + '_' + str(i)] = code[k:k+contigLength]
        k = k + contigLength



if __name__ == "__main__":
    # read input fasta file
    data_list = []
    first_line = True
    assembly_seq = ''
    primary_letter = ["A","C","G","T"]
    with open (inputfile,'r')as file:
        while True:
            line = file.readline().strip()
            if not line:
                break
            if first_line:
                assembly_header = line[1:]
                first_line =False
            elif line[0] == '>' :
                clean_save(assembly_header,assembly_seq.upper())
                assembly_header = line[1:]
                assembly_seq = ''
            else:
                assembly_seq = assembly_seq + line
    clean_save(assembly_header,assembly_seq.upper())
    number_assembly = len(data_list)
    # clean,split and encode assembly sequence
    all_code = []
    all_codeR = []
    mgr = Manager()
    segment_dic = mgr.dict()
    segment_dic_R = mgr.dict()
    pool = Pool(threads)
    pool.map(multi_encode, data_list)
    pool.close()
    pool.join()
    for i in range(number_assembly):
        for j in range(len(data_list[i][1])//contigLength):
            all_code.append(segment_dic[data_list[i][0] + '_' + str(j)])
            all_codeR.append(segment_dic_R[data_list[i][0] + '_' + str(j)])
    np.save( os.path.join(outDir, contigType +'_code'), np.array(all_code) )
    np.save( os.path.join(outDir, contigType +'_codeR'), np.array(all_codeR) )