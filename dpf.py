#### Step 1: import and parser ####
import os, sys, optparse, warnings
import time
import numpy as np
from Bio.Seq import Seq
from multiprocessing import Manager
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

print("1. Import Module Successfully.")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


prog_base = os.path.split(sys.argv[0])[1]
parser = optparse.OptionParser()
parser.add_option("-i", "--in", action = "store", type = "string", dest = "input_file",help = "input fasta file")
parser.add_option("-m", "--mod", action = "store", type = "string", dest = "modDir",default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"), help = "model directory (default ./models)")
parser.add_option("-o", "--out", action = "store", type = "string", dest = "output_dir", default='./', help = "output directory")
parser.add_option("-l", "--len", action = "store", type = "int", dest = "cutoff_len", default=1, help = "predict only for sequence >= L bp (default 1)")
parser.add_option("-s", "--seedscore", action = "store", type = "int", dest = "seed_score",default=0.8, help = "minimum score to be considered as seed")
parser.add_option("-n", "--seednumber", action = "store", type = "int", dest = "seed_number",default=4, help = "minimum number of seed, 1 seed = 3000bp")
parser.add_option("-e", "--extendscore", action = "store", type = "int", dest = "extend_score",default=0.5, help = "minimum score to extend on sequence")

    
(options, args) = parser.parse_args()
if (options.input_file is None) :
    sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
    filelog.write(prog_base + ": ERROR: missing required command-line argument")
    parser.print_help()
    sys.exit(0)

input_file = options.input_file
if options.output_dir != './' :
    output_dir = options.output_dir
else :
    output_dir = os.path.dirname(os.path.abspath(input_file))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cutoff_len = options.cutoff_len
seed_score = options.seed_score
seed_number = options.seed_number
extend_score = options.extend_score
contigLength = 3000

time_start = time.time()


#### Step 2: load model ####
modDir = options.modDir
print("   model directory {}".format(modDir))


modPattern = 'model_'
modName = [ x for x in os.listdir(modDir) if modPattern in x and x.endswith(".h5") ][0]
model = load_model(os.path.join(modDir, modName))
Y_pred_file = [ x for x in os.listdir(modDir) if modPattern in x and "Y_pred" in x ][0]
with open(os.path.join(modDir, Y_pred_file)) as f:
    tmp = [line.split() for line in f][0]
    Y_pred = [float(x) for x in tmp ]
Y_true_file = [ x for x in os.listdir(modDir) if modPattern in x and "Y_true" in x ][0]
with open(os.path.join(modDir, Y_true_file)) as f:
    tmp = [ line.split()[0] for line in f]
    Y_true = [ float(x) for x in tmp ]   
null = []
for i in range(len(Y_true)):
    if Y_true[i] == 0:
        null.append(Y_pred[i])


time_end = time.time()
print("2. Load Model Successfully.")
print('Time for loading model:',time_end-time_start)

# clean the output file
outfile = os.path.join(output_dir, os.path.basename(input_file)+'_dpfpred.txt')
predF = open(outfile, 'w')
writef = predF.write('@'.join(['name', 'start_base','end_base', 'score', 'pvalue'])+'\n')
predF.close()

outfile_boundary = os.path.join(output_dir, os.path.basename(input_file)+'_boundary.txt')
pred_boundary = open(outfile_boundary, 'w')
writef = pred_boundary.write('@'.join(['name', 'boundary'])+'\n')
pred_boundary.close()


outfile_phage_seq = os.path.join(output_dir, os.path.basename(input_file)+'_phage_seq.fasta')
pred_phage_seq = open(outfile_phage_seq, 'w')
pred_phage_seq.close()




#### Function ####
def clean_save(header,seq):
# split sequence when character is not A, T, C or G, sequence which length is less 3000 will be discarded
    i = 0
    j = 0
    start = 0
    end = 0
    temp = header.split()
    ID = temp[0]
    descriprion = ' '.join(temp[1:])
    while i < len(seq):
        if seq[i] not in primary_letter:
            if (i - start)>=3000:
                end = i
                data_list.append((ID + '_' + str(j),str(start),str(end-1),seq[start:end],descriprion))
                initial_site_seq[ID + '_' + str(j)] = start
                j = j + 1
            start = i + 1
        i = i + 1
    if (i - start)>=3000:
        end = i
        data_list.append((ID + '_' + str(j),str(start),str(end-1),seq[start:end],descriprion))
        initial_site_seq[ID + '_' + str(j)] = start
        

def encode(seq) :
# one-hot encoding
    values = list(seq)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(dtype='int32', sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def multi_encode_predict(data) :
# encode sequence, split, save encoding results and predict 
    global encode_time
    global pred_time
    encode_start_time = time.time()
    ID = data[0]
    seq = data[3]
    segment_num = len(seq)//3000
    seq_code=encode(seq)
    seqR = Seq(seq[:3000*segment_num]).reverse_complement()
    seq_codeR = encode(seqR)
    split_save(ID,seq_code,segment_num)
    split_save_R(ID,seq_codeR,segment_num)
    encode_end_time = time.time()
    encode_time = encode_time + (encode_end_time - encode_start_time)
    pred_start_time = time.time()
    pred_save(ID,segment_num)
    pred_end_time = time.time()
    pred_time = pred_time + (pred_end_time - pred_start_time)
    
def split_save(ID,code,segment_num) :
    for i in range(segment_num):
        segment_dic[ID + '_' + str(i)] = code[i*contigLength:(i+1)*contigLength]

def split_save_R(ID,code,segment_num) :
    for i in range(segment_num):
        segment_dic_R[ID + '_' + str(segment_num - i - 1)] = code[i*contigLength:(i+1)*contigLength]

def pred_save(ID,segment_num):
    for i in range(segment_num):
        segment_ID = ID + '_' + str(i)
        codefw = segment_dic[segment_ID]
        codebw = segment_dic_R[segment_ID]
        score = model.predict([np.array([codefw]), np.array([codebw])], batch_size=1)
        pvalue = sum([x>score for x in null])/len(null)
        dic_score[segment_ID] = score[0][0]
        dic_pvalue[segment_ID] = pvalue[0][0]


def find_phage(ID):
#define phage part based on dpf_score and parameters(seed_score, seed_number and extend_score)
    i = 0
    j = 0
    k = 0
    temp = []
    temp_dic = {}
    seed = []
    start = 0
    end = 0
    output = []
    flag = True
    while True:
        if (ID + '_' + str(i) + '_' + str(j)) in dic_score:
            temp_dic[k] = dic_score[ID + '_' + str(i) + '_' + str(j)]
            j = j + 1
            k = k + 1
        else:
            i = i + 1
            j = 0
            if (ID + '_' + str(i) + '_' + str(j)) not in dic_score:
                break
    k = 0
    while True:
        if k in temp_dic:
            if temp_dic[k] >= seed_score:
                temp.append(k)
            k = k + 1
        else:
            break
    for j in range(len(temp) - 1):
        if temp[j + 1] - temp[j] == 1:
            if flag:
                start = temp[j]
                end = temp[j + 1]
                flag = False
            else:
                end = temp[j + 1]
        else:
            if (end - start + 1) >= seed_number and (start,end) not in seed:
                seed.append((start,end))
            flag = True
    if (end - start + 1) >= seed_number and (start,end) not in seed:
        seed.append((start,end))
    for x in range(len(seed)):
        start = seed[x][0]
        end = seed[x][1]
        while (start - 1) in temp_dic :
            if temp_dic[start - 1] >= extend_score:
                start  = start - 1
            else:
                break
        while (end + 1) in temp_dic :
            if temp_dic[end + 1] >= extend_score:
                end  = end + 1
            else:
                break
        i = 0
        j = 0
        k = 0
        seq_start = ''
        seq_end = ''
        while True:
            if (ID + '_' + str(i) + '_' + str(j)) in dic_score:
                if k == start:
                    initial_site = initial_site_seq[ID + '_' + str(i)]
                    seq_start = initial_site + j*3000 + 1
                if k == end:
                    initial_site = initial_site_seq[ID + '_' + str(i)]
                    seq_end = initial_site + (j+1)*3000
                    break
                j = j + 1
                k = k + 1
            else:
                i = i + 1
                j = 0
                if (ID + '_' + str(i) + '_' + str(j)) not in dic_score:
                    break
        if seq_start != '' and seq_end != '':
            output.append((seq_start,seq_end))
    return np.unique(output)


#### Step 3: encode sequences in input fasta and predict scores ####


encode_time = 0
pred_time = 0
original_fasta = {}
data_list = []
initial_site_seq = {}
first_line = True
assembly_seq = ''
primary_letter = ["A","C","G","T"]
with open (input_file,'r')as file:
    while True:
        line = file.readline().strip()
        if not line:
            break
        if first_line:
            assembly_ID = line[1:]
            first_line =False
        elif line[0] == '>' :
            clean_save(assembly_ID,assembly_seq.upper())
            original_fasta[assembly_ID] = assembly_seq.upper()
            if len(data_list) >= 100:
                all_code = []
                all_codeR = []
                mgr = Manager()
                segment_dic = mgr.dict()
                segment_dic_R = mgr.dict()
                dic_score = mgr.dict()
                dic_pvalue = mgr.dict()
                for i in range(len(data_list)):
                    multi_encode_predict(data_list[i])
                with open(outfile,'a') as out_file:
                    for i in range(len(data_list)):
                        for j in range(len(data_list[i][3])//contigLength):
                            segment_ID = data_list[i][0] + '_' + str(j)
                            start = int(data_list[i][1])
                            score = str(dic_score[segment_ID])
                            pvalue = str(dic_pvalue[segment_ID])
                            start_base = str(3000*j + 1 + start)
                            end_base = str(3000*(j+1) + start)
                            out_file.write('@'.join([segment_ID, start_base, end_base, score, pvalue])+'\n')
                pred_boundary = open(outfile_boundary, 'a')
                pred_phage_seq = open(outfile_phage_seq, 'a')
                for i in range(len(data_list)):
                    temp = data_list[i][0].split('_')
                    ID = '_'.join(temp[:-1])
                    description = data_list[i][4]
                    last_temp = data_list[i-1][0].split('_')
                    if i != 0 and ID == '_'.join(last_temp[:-1]):
                        continue
                    phage_boundary = find_phage(ID).tolist()
                    if phage_boundary != []:
                        pred_boundary.write( ID + '@' + (','.join([str(i) for i in phage_boundary]))+'\n')
                        for j in range(0,len(phage_boundary),2):
                            pred_phage_seq.write('>' + ID + '_' + str(phage_boundary[j]) + '_' + str(phage_boundary[j + 1]) + ' ' +description + '\n')
                            pred_phage_seq.write(original_fasta[ID][phage_boundary[j] - 1:phage_boundary[j + 1]]+ '\n')
                pred_boundary.close()
                pred_phage_seq.close()
                original_fasta = {}
                data_list = []
                initial_site_seq = {}
            assembly_ID = line[1:]
            assembly_seq = ''
        else:
            assembly_seq = assembly_seq + line

clean_save(assembly_ID,assembly_seq.upper())
original_fasta[assembly_ID] = assembly_seq.upper()




if data_list != []:
    all_code = []
    all_codeR = []
    mgr = Manager()
    segment_dic = mgr.dict()
    segment_dic_R = mgr.dict()
    dic_score = mgr.dict()
    dic_pvalue = mgr.dict()
    for i in range(len(data_list)):
        multi_encode_predict(data_list[i])
    with open(outfile,'a') as file:
        for i in range(len(data_list)):
            for j in range(len(data_list[i][3])//contigLength):
                segment_ID = data_list[i][0] + '_' + str(j)
                start = int(data_list[i][1])
                score = str(dic_score[segment_ID])
                pvalue = str(dic_pvalue[segment_ID])
                start_base = str(3000*j + 1 + start)
                end_base = str(3000*(j+1) + start)
                file.write('@'.join([segment_ID, start_base, end_base, score, pvalue])+'\n')
    pred_boundary = open(outfile_boundary, 'a')
    pred_phage_seq = open(outfile_phage_seq, 'a')
    for i in range(len(data_list)):
        temp = data_list[i][0].split('_')
        ID = '_'.join(temp[:-1])
        description = data_list[i][4]
        last_temp = data_list[i-1][0].split('_')
        if i != 0 and ID == '_'.join(last_temp[:-1]):
            continue
        phage_boundary = find_phage(ID).tolist()
        if phage_boundary != []:
            pred_boundary.write( ID + '@' + (','.join([str(i) for i in phage_boundary]))+'\n')
            for j in range(0,len(phage_boundary),2):
                pred_phage_seq.write('>' + ID + '_' + str(phage_boundary[j]) + '_' + str(phage_boundary[j + 1]) + ' ' +description + '\n')
                pred_phage_seq.write(original_fasta[ID][phage_boundary[j] - 1:phage_boundary[j + 1]]+ '\n')
    pred_boundary.close()
    pred_phage_seq.close()
print("3.Load, Clean, Encode, Predict Imported Sequences Successfully.")
print('Time for encoding:',encode_time)
print('Time for predicting:',pred_time)