# this script is to perform prediction of protein 2nd structure from nmr chemical shift data on NMRSTAR file
# the main part is from 'eva_pred_2nd_structure_v02_1.py
# the algorithm is based on LSTM model
# use regularization to overcome overfitting
# 
# use sklearn.utils.shuffle to random arrange numpy data firstly.
# 
# use 7 physcichemical properties and 6 chemical shift values as LSTM features.
# use state probability in 5-aa, 3-aa, and 1-aa pieces, together with LSTM outputs as FC features
# 
# the prediction result is save in prediction.txt, in which each protein has 4 lines
# 
# Use CPU not GPU
# 
# Usage: python pred_on_nmrstar_v2.py xxx.NMRSTAR_file


import def_protein_record_v6_0
import os
import dill
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import argparse
import fnmatch
from def_protein_record_v6_0 import read_state_prob_file
from statistics import mean
import re


parser = argparse.ArgumentParser(description='Predict protein secondary structure based on NMR chemical shift')
parser.add_argument('star_file_name', metavar='File', help='file name of NMRSTAR')
parser.add_argument('--resume', default='dir_ckpt/FC2_200_lr0001_wd00075_ep300_8853.ckpt', type=str, metavar='FILE', help='path to latest checkpoint')
parser.add_argument('--gpu_id', default=1, type=int, help='GPU id to use.')
parser.add_argument('--batch_size', default=4096, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--layer_fc', default=2, type=int, metavar='N',help='layers of full connection')
parser.add_argument('--node_fc2', default=200, type=int, metavar='N',help='nodes of 2nd FC layer')


def main():
    
    args = parser.parse_args()
    star_file_name = args.star_file_name

    size_slicing = 11
    size_padding = size_slicing // 2    
    

#####################################################
####   check dir and file
#####################################################

    file_mean_std_feature = 'statistical_data_for_pred/mean_std_feature.csv'
    file_max_min_feature = 'max_min_feature.csv'

    file_state_prob_5aa = 'statistical_data_for_pred/quintuplet_state_prob.csv'
    file_state_prob_3aa = 'statistical_data_for_pred/triplet_state_prob.csv'
    file_state_prob_1aa = 'statistical_data_for_pred/singlet_state_prob.csv'


    file_ckpt = args.resume


    if not os.path.isfile(file_ckpt):
        print('can not find ckpt data file')
        exit()

    if not os.path.isfile(file_mean_std_feature):
        print('miss mean_std_feature file')
        exit()
    

    if not os.path.isfile(file_state_prob_5aa):
        print('miss state prob 5aa file')
        exit()

    if not os.path.isfile(file_state_prob_3aa):
        print('miss state prob 3aa file')
        exit()

    if not os.path.isfile(file_state_prob_1aa):
        print('miss state prob 1aa file')
        exit()


# ######################################################
##   read  data preparation parameters
# ######################################################

# read state probability of aa pieces

    dict_state_prob_5aa = read_state_prob_file(file_state_prob_5aa)
    dict_state_prob_3aa = read_state_prob_file(file_state_prob_3aa)
    dict_state_prob_1aa = read_state_prob_file(file_state_prob_1aa)


    # read scaling parameters

    with open(file_mean_std_feature, 'r') as f_reader:
        lines = f_reader.readlines()

    if len(lines) != 17:
        print('something wrong in your mean_std_feature file')
        exit()


    mean_feature = []
    std_feature = []
    for line in lines[1:]:
        temp = line.strip().split(',')
        mean_feature.append(float(temp[0]))
        std_feature.append(float(temp[1]))



#####################################################
####   read prediction model
#####################################################

    # the following parameters MUST be SAME to the trained model
    # Also, you need check the model class defination behind this main() function, to make sure that the model structure is same 

    num_LSTM_features = 35   # 22 aa name in onehot, 7 physichemical properties, 6 cs
    Batch_Size = args.batch_size           # default 4096

    Input_Size = num_LSTM_features
    Seq_Length = 11    # size of windowing is 11 a.a. length
    Hidden_Size = 128
    Output_Size = 3
    Num_LSTM_Layers = 2
    Num_LSTM_Direction = 2  # 2: if bidirectional; 1: monodirectional
    Num_FC_Layers = args.layer_fc 
    Using_prob_state = True
    Node_FC2 = args.node_fc2

    # device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("using CPU ")


    # Initiate NN model
    model = LSTM_protein_classifier(Batch_Size, Input_Size, Hidden_Size, Seq_Length, Output_Size, Num_LSTM_Layers, Num_LSTM_Direction, Num_FC_Layers, Node_FC2, Using_prob_state)
    model = model.to(device)

    # # read checkpoint to initiate model
    # if args.gpu_id is None:
    #     checkpoint = torch.load(args.resume)
    # else:
    #     # Map model to be loaded to specified single gpu.
    #     loc = 'cuda:{}'.format(args.gpu_id)
    #     checkpoint = torch.load(file_ckpt, map_location=loc)

    checkpoint = torch.load(args.resume, map_location = device)

    model.load_state_dict(checkpoint)
    print("=> loaded checkpoint '{}'".format(args.resume))


    # switch to evaluate mode
    model.eval()

    
    #####################################################
    ####   read protein data
    #####################################################


    list_state_label = []
    list_state_pred = []
    list_acc = []
    list_pdbid = []
    list_aa_name = []


    protein = def_protein_record_v6_0.protein_v2('0', '0')
    temp = protein.read_star(star_file_name)

    if temp != 1:
        print('error in reading nmrstar file')
        exit()

    # protein.disp()

    protein.padding( size_padding )
    list_peptide_temp = protein.slicing( size_slicing )

    list_mat_peptide = []
    list_prob_state = []
    list_state = []

    # convert protein data format to what model prediction need
    for peptide_temp in list_peptide_temp:
        peptide_temp.update_state_prob(dict_state_prob_5aa, len_piece=5)
        peptide_temp.update_state_prob(dict_state_prob_3aa, len_piece=3)
        peptide_temp.update_state_prob(dict_state_prob_1aa, len_piece=1)
        mat_peptide, prob_state, state = peptide_temp.to_numpy_tuple()
        mat_peptide = mat_peptide[:, :-3]   # remove acc and 2 torsion angles

        for idx in range(1, 14):
            idx_feature = idx - 1
            mat_peptide[:, idx] = ( mat_peptide[:, idx] - mean_feature[ idx_feature ] ) / std_feature[ idx_feature ]
            # mat_peptide[:, idx] = ( mat_peptide[:, idx] - min_feature[ idx_feature ] ) / ( max_feature[ idx_feature ] - mat_peptide[:, idx] )

        # one-hot coding for amino acid name
        aa_seq = mat_peptide[:, 0]
        aa_seq = aa_seq.reshape(-1,1)
        aa_seq = torch.LongTensor(aa_seq)
        aa_seq_onehot = torch.zeros(aa_seq.shape[0], 22)
        aa_seq_onehot = aa_seq_onehot.scatter_(1, aa_seq, 1)
        aa_seq_onehot = aa_seq_onehot.numpy()

        mat_peptide = np.concatenate(  (aa_seq_onehot, mat_peptide[:,1:]), axis=1 )
        
        
        list_mat_peptide.append(mat_peptide)
        list_prob_state.append(prob_state)
        list_state.append(state)


    list_mat_peptide = np.stack(list_mat_peptide)
    list_prob_state = np.stack(list_prob_state)
    list_state = np.stack(list_state)

    batch_peptide_features = torch.from_numpy(list_mat_peptide)
    batch_prob_state = torch.from_numpy(list_prob_state)
    batch_state_label = torch.from_numpy(list_state)

    softmax = torch.nn.Softmax(dim = 1)


    with torch.no_grad():
        batch_peptide_features = batch_peptide_features.to(device)
        batch_peptide_features = torch.transpose(batch_peptide_features, 0, 1)

        batch_prob_state = batch_prob_state.to(device)

        model_out = model(batch_peptide_features, batch_prob_state)

        prob_pred = softmax(model_out)
        _, pred = torch.max(prob_pred,dim=1)

    pred = pred.cpu().numpy()
    
    protein.unpadding( size_padding )
    aa_seq = protein.aa_seq

    # print(len(aa_seq))
    # print(aa_seq)

    # print(len(pred))
    # print(pred)

    for index, (aa, ss) in enumerate( zip(aa_seq, pred) ):
        ss = def_protein_record_v6_0.second_structure_uncode(ss)
        print("{}: {} - {}".format(index, aa, ss))









### ~~~~~~~~~~~~~~~   Class Defination     ~~~~~~~~~~~~~~~~~~~~    #######

# the shape of input numpy array is [n_samples, len_seq,  n_feature] (XXX, 11, 36)

class Protein2ndStruDataset(Dataset):
    def __init__(self, list_train_peptide_features, list_train_prob_state, list_train_state_label, transform=None):

        self.list_train_peptide_features = list_train_peptide_features  
        self.list_train_prob_state = list_train_prob_state
        self.list_train_state_label = list_train_state_label 
        self.transform = transform
        
    
    def __len__(self):
        return self.list_train_peptide_features.shape[0] # the number_samples is the second dimension
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        peptide_features = self.list_train_peptide_features[idx, :, :]
        prob_state = self.list_train_prob_state[idx, :]
        state_label = self.list_train_state_label[idx]

        state_label = state_label.astype(int)

        # if self.transform:
        #     aa_seq_piece = self.transform(aa_seq_piece)

        return(peptide_features, prob_state, state_label)


# define neural network model
class LSTM_protein_classifier(torch.nn.Module):

    def __init__(self, Batch_Size, Input_Size, Hidden_Size, Seq_Length, Output_Size, Num_LSTM_Layers, Num_Direction, Num_FC_Layers, Node_FC2, Using_prob_state=True):
        super(LSTM_protein_classifier, self).__init__()
        self.Batch_Size = Batch_Size
        self.Hidden_Size = Hidden_Size
        self.Input_Size = Input_Size
        self.Seq_Length = Seq_Length
        self.Output_Size = Output_Size
        self.Num_Direction = Num_Direction
        self.Num_LSTM_Layers = Num_LSTM_Layers
        self.Num_FC_Layers = Num_FC_Layers
        self.Using_prob_state = Using_prob_state
        self.Node_FC2 = Node_FC2
 
        if Num_Direction == 2:
            flag_bidirect = True
        else:
            flag_bidirect = False
        
        # using probability of state in 5, 3, 1-aa pieces. Or not
        if self.Using_prob_state == True:
            Input_Size_FC1 = Hidden_Size * Num_Direction * Seq_Length + 9
        else:
            Input_Size_FC1 = Hidden_Size * Num_Direction * Seq_Length
                 
        Output_Size_FC1 = Output_Size
        Output_Size_FC2 = Output_Size
        Output_Size_FC3 = Output_Size
        
        if Num_FC_Layers == 2:
            Output_Size_FC1 = Node_FC2
        if Num_FC_Layers == 3:   
            Output_Size_FC1 = Node_FC2
            Output_Size_FC2 = 20
        
        Input_Size_FC2 = Output_Size_FC1
        Input_Size_FC3 = Output_Size_FC2

        self.lstm = torch.nn.LSTM(input_size=Input_Size, hidden_size=Hidden_Size, num_layers=Num_LSTM_Layers, bidirectional=flag_bidirect)
        
        
        self.fc1 = torch.nn.Linear(Input_Size_FC1, Output_Size_FC1)
        self.fc2 = torch.nn.Linear(Input_Size_FC2, Output_Size_FC2)
        self.fc3 = torch.nn.Linear(Input_Size_FC3, Output_Size_FC3)

        self.relu = torch.nn.ReLU()


        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    # the shape of aa_segment should be [seq_length, batch_size, feature_size]
    # the shape of prob_state should be [batch_size, 9]
    def forward(self, aa_segment, prob_state=None):

        # print(aa_segment.dtype, prob_state.dtype)

        # the shape of lstm_out is [seq_length, batch_size, hidden_size * num_direction]
        lstm_out, _ = self.lstm(aa_segment)
        
        # batch_size of the last batch may not the defined one 
        batch_size = aa_segment.shape[1]

        # the shape of lstm is [batch_size, seq_length, hidden_size * num_direction]
        lstm_out = lstm_out.permute(1, 0, 2)

        # reshape lstm_out to [batch_size, Hidden_Size * num_direction * seq_len]
        lstm_out = lstm_out.reshape(batch_size, -1)

        # if using probability of state as input feature of aa segment, it should be concatenate into LSTM output
        # then be put into next layer
        if self.Using_prob_state == True:
            if None == prob_state or prob_state.shape[1] != 9:
                print('bad probability of state, please check it')
                exit()
            
            # print(lstm_out.dtype, prob_state.dtype)
            
            lstm_out = torch.cat( (lstm_out, prob_state), 1 )

        if self.Num_FC_Layers == 1:
            fc1_out = self.fc1(lstm_out)
            # fc1_out = self.relu1(fc1_out)
            return(fc1_out)
        elif self.Num_FC_Layers == 2:
            fc1_out = self.fc1(lstm_out)
            fc1_out = self.relu(fc1_out)
            fc2_out = self.fc2(fc1_out)
            # fc2_out = self.relu2(fc2_out)
            return(fc2_out)
        else:
            fc1_out = self.fc1(lstm_out)
            fc1_out = self.relu(fc1_out)
            fc2_out = self.fc2(fc1_out)
            fc2_out = self.relu(fc1_out)
            fc3_out = self.fc3(fc2_out)
            # fc3_out = self.relu3(fc3_out)
            return(fc3_out)




if __name__ == '__main__':
    main()



