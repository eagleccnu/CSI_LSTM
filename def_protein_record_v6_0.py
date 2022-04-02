import pynmrstar
import json
import os
import fnmatch
import re
from re import findall
from difflib import SequenceMatcher
from pprint import pprint
import dill
import numpy as np
from statistics import mean
import pandas as pd


# read a nmrstar file, and tell whether the sample of NMR experiment is single entity
def tell_single_entity_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star)
    # entity_loop = entry.get_loops_by_category("Entity_assembly")
    entity_ID_array = entry.get_tag('Entity_assembly.Entity_ID')
    num_entity = len(entity_ID_array)
    # print('The number of entity is %d' % num_entity)
    if num_entity == 1:
        # print('single entity')
        return(True)
    else:
        # print('multiple entity')
        return(False)



# read a nmrstar file, and tell whether the entry has only one cs set
def tell_single_cs_set_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star)
    chemical_shift_loops = entry.get_loops_by_category("Atom_chem_shift")
    num_cs_set = len(chemical_shift_loops)
    if num_cs_set == 1:
        return(True)
    else:
        print("num of cs_set is {}".format(num_cs_set))
        return(False)


# after tell the nmrstar file contains just only one cs_set
# read cs set into a pandas framework from nmrstar file
def read_cs_into_pd_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star, convert_data_types=True)
    chemical_shift_loops = entry.get_loops_by_category("Atom_chem_shift")
    chemical_shift_loop = chemical_shift_loops[0]
    # tags = chemical_shift_loop.get_tag_names()
    # print(tags)
    cs_set = chemical_shift_loop.get_tag(['ID', 'Comp_index_ID','Seq_ID', 'Comp_ID', 'Atom_ID', 'Val'])
    pd_cs_set = pd.DataFrame.from_records(cs_set,columns=['ID', 'Comp_index_ID','Seq_ID', 'Comp_ID', 'Atom_ID', 'Val'])
    
    return(pd_cs_set)


# input is a pandas framework which is output of read_cs_into_pd_from_star
# return a tuple of amino acid residue sequence, and list of chemical shift of each residue
def read_cs_from_pd(pd_cs_set):
    aa_seq = []
    cs_dict_seq = []

    # divide cs_set pandas into subsets by groupby on 'comp_index_ID'
    # each subset is one aa
    grp_by_comp = pd_cs_set.groupby('Comp_index_ID')

    for idx_1, aa in grp_by_comp:
        temp = aa.iloc[0]
        aa_seq.append(temp['Comp_ID'])

        dict_atom_cs = {}
        for idx_2, row in aa.iterrows():
            dict_atom_cs[row['Atom_ID']] = float(row['Val'])
    
        cs_dict_seq.append(dict_atom_cs)

    aa_seq = [abbr_reduce(i) for i in aa_seq]
    
    return(aa_seq, cs_dict_seq)
    

# read aa seq of protein from nmrstar file
def read_seq_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star)
    seq_one_letter_code = entry.get_tag('Entity.Polymer_seq_one_letter_code')
    if len(seq_one_letter_code) == 0:
        return(None)
    else:
        aa_seq = seq_one_letter_code[0]
        aa_seq = aa_seq.replace('\n','')
        aa_seq = aa_seq.replace('\r','')
        aa_seq = list(aa_seq)
    
    return(aa_seq)



# def get_cs_from_nmrstar(file_nmrstar):
#     entry = pynmrstar.Entry.from_file(file_nmrstar)
#     entry.print_tree()
#     assembly = entry.get_saveframe_by_name('assembly')
#     assembly.print_tree()
#     entity = assembly.get_loop_by_category("Entity_assembly")
#     print(entity[0])

# pass


# read prediction result from CSI 2.0 website
# given pdbid and dir_csi_pred
# return a tuple of (list_slice_begin, list_slice_end, list_slice_aa, list_slice_state)
def read_csi_pred(pdbid, dir_csi_pred):

    dict_state_name_trans = {}
    dict_state_name_trans['H'] = 'H'
    dict_state_name_trans['B'] = 'E'
    dict_state_name_trans['C'] = 'C'


    if not os.path.isdir(dir_csi_pred):
        print('lack prediction result dir')
        return(None)
    
    list_all_files = os.listdir(dir_csi_pred)
    matched_list = fnmatch.filter(list_all_files, '%s_*' % pdbid)

    if len(matched_list) != 1:
        print('Something is wrong with this PDB_ID')
        return(None)

    file_name = dir_csi_pred + matched_list[0]

    list_slice_begin = [] 
    list_slice_end = []
    list_slice_aa = []
    list_slice_state = []

    with open(file_name, 'r') as file_reader:

        flag_slice = 1   # flag that, we read a new slice
        line = file_reader.readline()
        
        while line != '':
            # print(line)

            if flag_slice == 1:   # what read is a new slice
                slice_begin = re.findall('^([0-9]+)\s', line) 
                if len(slice_begin) != 0:
                    # print(slice_begin[0])
                    list_slice_begin.append( int(slice_begin[0]) )
                    slice_end = re.findall('\s([0-9]+)', line) 
                    if len(slice_end) != 0:
                        list_slice_end.append( int(slice_end[0]) )
                    else:
                        return(None)
                    slice_aa = re.findall('([A-Za-z]+)', line)
                    if len(slice_aa) != 0:
                        list_slice_aa.append( slice_aa[0] )
                    else:
                        return(None)
                    flag_slice = 0 # next line still is this slice
            else:   # what's read is old slice
                slice_state = re.findall('([A-Za-z]+)', line) 
                if len(slice_begin) != 0:
                    list_slice_state.append( slice_state[0] )
                else:
                    return(None)
                flag_slice = 1   # reset this flag    
            
            line = file_reader.readline()

    lists_slice_aa = []
    for seg in list_slice_aa:
        list_temp = [aa for aa in seg]
        lists_slice_aa.append(list_temp.copy())
    
    lists_slice_state = []
    for seg in list_slice_state:
        list_temp = [dict_state_name_trans[ state ] for state in seg]    # convert 'B' to 'E' by dict
        lists_slice_state.append(list_temp.copy())


    return(list_slice_begin, list_slice_end, lists_slice_aa, lists_slice_state)



# read prediction result from CSI 3.0 website
# given pdbid and dir_csi_pred
# return a tuple of (list_index, list_aa, list_state)
def read_csi30_pred(pdbid, dir_csi_pred):

    dict_state_name_trans = {}
    dict_state_name_trans['H'] = 'H'
    dict_state_name_trans['B'] = 'E'
    dict_state_name_trans['C'] = 'C'


    if not os.path.isdir(dir_csi_pred):
        print('lack prediction result dir')
        return(None)
    
    list_all_files = os.listdir(dir_csi_pred)
    matched_list = fnmatch.filter(list_all_files, '%s_*' % pdbid)

    if len(matched_list) != 1:
        print('Something is wrong with this PDB_ID')
        return(None)

    file_name = dir_csi_pred + matched_list[0]

    list_index = []
    list_aa = []
    list_state = []

    with open(file_name, 'r') as file_reader:

        line = file_reader.readline()
        
        while line != '':
            # print(line)
            index = re.findall('^([0-9]+)\s', line)
            if len(index) == 1:
                list_index.append(int(index[0]))
                aa = re.findall('^[0-9]+\s([A-Z]+)\s', line)
                if len(aa) == 1:
                    list_aa.append(aa[0])
                else:
                    return(None)
                state = re.findall('^[0-9]+\s[A-Z]+\s([A-Z]{1})\s', line)
                if len(state) == 1:
                    list_state.append(state[0])
                else:
                    print('something wrong in reading state')
                    return(None)

            line = file_reader.readline()
    
    list_aa = [ abbr_reduce(aa) for aa in list_aa ]
    list_state_conv = [dict_state_name_trans[state] for state in list_state]

    # for index, aa, state, state_conv in zip(list_index, list_aa, list_state, list_state_conv):
    #     print(index, aa, state, state_conv)

    return(list_index, list_aa, list_state_conv)


# read a dssp file
# return a tuple of amino acid residue sequence, 2nd structure sequence, ACC, and torsion angle phi and psi
def read_dssp_v2(file_name):


    # print(file_name)

    if os.path.isfile(file_name):
        # print("it's here")
        aa_seq = []
        second_structure_seq = []
        acc_seq = []
        phi_seq = []
        psi_seq = []

        with open(file_name,'r') as read_file:
        
            flag_block = 0
            line_temp = read_file.readline()

            while line_temp:
                if flag_block == 0 and bool(findall("#  RESIDUE AA",line_temp)):
                    flag_block = 1
                    # print(line_temp)
                elif flag_block == 1:
                    # print(line_temp[13], line_temp[16])
                    aa_seq.append(line_temp[13])
                    second_structure_seq.append(line_temp[16])

                    acc_str = line_temp[35:38]
                    acc = int(acc_str)
                    acc_seq.append(acc)
                    
                    phi_str = line_temp[103:109]
                    phi_seq.append(float(phi_str))

                    psi_str = line_temp[109:115]
                    psi_seq.append(float(psi_str))
                else:
                    pass
                line_temp = read_file.readline()
        
        # print(aa_seq)
        # print(second_structure_seq)
        return(aa_seq, second_structure_seq, acc_seq, phi_seq, psi_seq)



# dssp file used ONE letter code for amino acid abbreviation
# and mark C/CYS as 'a'/'b'/'c' and S-S bond pair
# this function is to replace 'a'/'b'/'c' with 'C' in amino acid sequence of abbreviation
def fix_dssp_abbr(aa_seq):
    aa_str = ''.join(map(str, aa_seq))  # convert list to str, for application of re
    fixed = re.sub('[a-z]', 'C', aa_str)
    fixed = list(fixed)
    return(fixed)


# this function is to list all file name of XXXX_i.cs 
def find_cs_file_v2(dir_ChemShift, bmrbid):
    list_all_cs_files = os.listdir(dir_ChemShift)
    matched_list = fnmatch.filter(list_all_cs_files, '%s_*' % bmrbid)
    return matched_list




# read a ChemShift file
# return a tuple of amino acid residue sequence, and list of chemical shift of each residue
def read_cs(file_name):
    if os.path.isfile(file_name):
        print("it's here")

        aa_seq_2 = []
        chem_shift_seq = []  # a list contain atom_cs dictionay

        with open(file_name,'r') as read_file:
            flag_block = 0
            flag_atom_cs = 0
            index_aa_seq = 0  # initial position in the aa seq.
            dict_atom_cs = {} # an empty dictionary will contain atom:cs_value pair
            
            line_temp = read_file.readline()

            while line_temp:
                # it is not a block
                if flag_block == 0 and ("loop_" not in line_temp):
                    line_temp = read_file.readline()
                    continue

                # start of a block
                elif flag_block == 0 and ("loop_" in line_temp):
                    flag_block = 1
                    # print(line_temp)
                    line_temp = read_file.readline()
                    continue

                # end of a block
                elif flag_block == 1 and ("stop_" in line_temp):
                    flag_block = 0
                    # it's the end of cs block
                    if flag_atom_cs == 1:
                        flag_atom_cs = 0
                        if len(aa_seq_2) > len(chem_shift_seq):
                            chem_shift_seq.append(dict_atom_cs)    # Save the atom_cs pair of the last residue
                    line_temp = read_file.readline()
                    continue

                # inside of a block
                elif flag_block == 1 and ("stop_" not in line_temp):

                    # this is not a ATOM chem shift block
                    if flag_atom_cs == 0 and ("_Atom_chem_shift" not in line_temp):
                        line_temp = read_file.readline()
                        continue

                    # this is the beginning of ATOM chem shift block
                    elif flag_atom_cs == 0 and ("_Atom_chem_shift" in line_temp):
                        # print(line_temp)
                        flag_atom_cs = 1
                        line_temp = read_file.readline()
                        continue

                    # this is inside of ATOM chem shift block
                    # split the line by blank, if the length of splitted list is 24
                    # then this line contains chem shift of one atom
                    elif flag_atom_cs == 1:
                        elements_splitted = line_temp.split()
                        if len(elements_splitted) == 24:
                            # the 5th element in each line is the resudue position in aa sequence
                            position_aa = elements_splitted[5]
                            # the 6th element is the aa resudue name
                            name_aa = elements_splitted[6]
                            # the 7th element in each line is atom_ID
                            atom_ID = elements_splitted[7]
                            # the 10th element is the chemical shift value
                            cs_value = float(elements_splitted[10])

                            # print(name_aa)

                            # this is the 1st residue
                            if index_aa_seq == 0:
                                index_aa_seq = position_aa
                                aa_seq_2.append(name_aa)  # the 1st aa residue name
                                dict_atom_cs[atom_ID] = cs_value
                            # inside a residue
                            if (index_aa_seq != 0) and (index_aa_seq == position_aa):
                                dict_atom_cs[atom_ID] = cs_value
                            # index != position, means this is the beginning of a new residue
                            if (index_aa_seq != 0) and (index_aa_seq != position_aa):
                                chem_shift_seq.append(dict_atom_cs)  # save atom_cs pair
                                index_aa_seq = position_aa  # update index
                                aa_seq_2.append(name_aa) 
                                del dict_atom_cs
                                dict_atom_cs = {}
                                dict_atom_cs[atom_ID] = cs_value
                            
                        line_temp = read_file.readline()
                        continue

                    else:
                        line_temp = read_file.readline()
                        continue
                
                else:
                    line_temp = read_file.readline()
        
        # print(chem_shift_seq)
        return(aa_seq_2, chem_shift_seq)

    else:
        return(None)


# In chem shift files, the aa residues are represented by THREE-letter abbreviation
# While in dssp files, they are represented by ONE-letter abbreviation
# this function is to reduce 3-letter abbreviation to 1-letter abbreviation
def abbr_reduce(long_abbr):
    dict_abbr_reduce = {}
    dict_abbr_reduce['ALA'] = 'A'
    dict_abbr_reduce['ARG'] = 'R'
    dict_abbr_reduce['ASN'] = 'N'
    dict_abbr_reduce['ASP'] = 'D'
    dict_abbr_reduce['CYS'] = 'C'
    dict_abbr_reduce['GLU'] = 'E'
    dict_abbr_reduce['GLN'] = 'Q'
    dict_abbr_reduce['GLY'] = 'G'
    dict_abbr_reduce['HIS'] = 'H'
    dict_abbr_reduce['ILE'] = 'I'
    dict_abbr_reduce['LEU'] = 'L'
    dict_abbr_reduce['LYS'] = 'K'
    dict_abbr_reduce['MET'] = 'M'
    dict_abbr_reduce['PHE'] = 'F'
    dict_abbr_reduce['PRO'] = 'P'
    dict_abbr_reduce['SER'] = 'S'
    dict_abbr_reduce['THR'] = 'T'
    dict_abbr_reduce['TRP'] = 'W'
    dict_abbr_reduce['TYR'] = 'Y'
    dict_abbr_reduce['VAL'] = 'V'

    list_long_abbr = list(dict_abbr_reduce.keys())

    if long_abbr in list_long_abbr:
        return(dict_abbr_reduce[long_abbr])
    else:
        return('X')



def aa_code(aa_char):
    dict_aa_code = {}
    dict_aa_code['A'] = 0
    dict_aa_code['C'] = 1
    dict_aa_code['D'] = 2
    dict_aa_code['E'] = 3
    dict_aa_code['F'] = 4
    dict_aa_code['G'] = 5
    dict_aa_code['H'] = 6
    dict_aa_code['I'] = 7
    dict_aa_code['K'] = 8
    dict_aa_code['L'] = 9
    dict_aa_code['M'] = 10
    dict_aa_code['N'] = 11
    # dict_aa_code['O'] = 12
    dict_aa_code['P'] = 12
    dict_aa_code['Q'] = 13
    dict_aa_code['R'] = 14
    dict_aa_code['S'] = 15
    dict_aa_code['T'] = 16
    # dict_aa_code['U'] = 18
    dict_aa_code['V'] = 17
    dict_aa_code['W'] = 18
    dict_aa_code['Y'] = 19
    dict_aa_code['X'] = 20
    dict_aa_code['B'] = 21

    return(dict_aa_code[aa_char])



def second_structure_code(second_structure_char):
    dict_2nd_structure_code = {}
    dict_2nd_structure_code['H'] = 0
    dict_2nd_structure_code['E'] = 1
    dict_2nd_structure_code['C'] = 2

    return(dict_2nd_structure_code[second_structure_char])


def second_structure_uncode(second_structure_code):
    dict_2nd_structure_char = {}
    dict_2nd_structure_char[0] = 'H'
    dict_2nd_structure_char[1] = 'E'
    dict_2nd_structure_char[2] = 'C'

    return(dict_2nd_structure_char[second_structure_code])




# the meaning of this phys_chem properties refer to Journal of Molecular Modeling 2001, 7 (9), 360-369.
def phys_chem_code(aa_char):
    dict_phys_chem_code = {}
    dict_phys_chem_code['A'] = [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23]
    dict_phys_chem_code['G'] = [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15]
    dict_phys_chem_code['V'] = [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49]
    dict_phys_chem_code['L'] = [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31]
    dict_phys_chem_code['I'] = [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45]
    dict_phys_chem_code['F'] = [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38]
    dict_phys_chem_code['Y'] = [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41]
    dict_phys_chem_code['W'] = [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42]
    dict_phys_chem_code['T'] = [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36]
    dict_phys_chem_code['S'] = [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28]
    dict_phys_chem_code['R'] = [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25]
    dict_phys_chem_code['K'] = [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27]
    dict_phys_chem_code['H'] = [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30]
    dict_phys_chem_code['D'] = [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20]
    dict_phys_chem_code['E'] = [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21]
    dict_phys_chem_code['N'] = [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22]
    dict_phys_chem_code['Q'] = [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25]
    dict_phys_chem_code['M'] = [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32]
    dict_phys_chem_code['P'] = [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34]
    dict_phys_chem_code['C'] = [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41]
    dict_phys_chem_code['X'] = [2.27, 0.17, 3.76, 0.48, 6.22, 0.28, 0.31]    # the 'wild card' a.a. owns averaged value
    dict_phys_chem_code['B'] = [0.00, 0.00, 0.00, 0.00, 6.00, 0.00, 0.00]    # a virtual 'blank' a.a. 

    return(dict_phys_chem_code[aa_char])



# read peptide state probability file, and return the dictionary of state probability of peptides.
def read_state_prob_file(file_state_prob):
    dict_state_prob = {}
    with open(file_state_prob) as file_reader:
        line = next(file_reader)  # drop the 1st line
        for line in file_reader:
            list_line_content = line.split()
            dict_state_prob[list_line_content[0]] = [float(list_line_content[5]), float(list_line_content[6]), float(list_line_content[7])]

    return(dict_state_prob)



# to calculate the average chem shift of a set of same type of amino acid
# In calculation, the unmeasured chemical shift values which marked with -100, are ignored
# the shape of cs_aa is a numpy array [num_aa, num_nuclei(6)]
# the shape of output is [num_nuclei]
def average_cs(cs_aa):
    cs_aa = np.array(cs_aa)
    num_piece = cs_aa.shape[0]
    num_nuclei = cs_aa.shape[1]

    cs_average = num_nuclei * [0.0]

    for col in range(num_nuclei):
        cs_vals = [cs for cs in cs_aa[:, col] if cs > -90]
        if len(cs_vals) == 0:
            cs_average[col] = -100    # which means the cs of this nuclei are not measured at all
        else:
            cs_average[col] = mean(cs_vals)
    
    return(cs_average)




# each a.a. residue has some NMR chemical shift data
# and those chemical shift data compose chemical shift feature
# used for deep learnding to tell the second structure of this a.a. residue
# the atom type (HN, CA, etc) is chosen from atom_ID.csv generated before.
# only 6 backbone chemical shifts are retained from previous version

class aa_chem_shift_v2:
    # cs_H = -100.0         # H atom, the default CS value of H is -10.0
    # cs_HA = -100.0         
    # cs_N = -100.0         # N atom, the default CS value of N is -100.0
    # cs_C = -100.0
    # cs_CA = -100.0        # C atom, the default CS value of C is -100.0
    # cs_CB = -100.0


    def __init__(self):
        self.cs_H = -100.0         
        self.cs_HA = -100.0        
        self.cs_N = -100.0
        self.cs_CA = -100.0
        self.cs_CB = -100.0
        self.cs_C = -100.0
    
    def set_cs(self, atom_ID, cs_value):
        if atom_ID == 'H':
            self.cs_H = cs_value
        elif atom_ID == 'HA':
            self.cs_HA = cs_value
        elif atom_ID == 'N':
            self.cs_N = cs_value
        elif atom_ID == 'C':
            self.cs_C = cs_value
        elif atom_ID == 'CA':
            self.cs_CA = cs_value
        elif atom_ID == 'CB':
            self.cs_CB = cs_value
        else:
            pass
    
    def disp(self):
        print("%3.1f\t %3.1f\t %3.1f\t %3.1f\t %3.1f\t %3.1f" % (self.cs_H, self.cs_HA, self.cs_N, self.cs_C, self.cs_CA, self.cs_CB))


class amino_acid:
    # name must be a char represents amino acid
    def __init__(self, name):
        self.name = name
        self.state_8 = ''
        self.state_3 = ''
        self.chemical_shift = aa_chem_shift_v2()
        self.acc = None
        self.phi = None
        self.psi = None


    
    # state is the secondary structure code readed out from dssp file 
    def set_state_8(self, state):
        self.state_8 = state
    
    # state is 'H' for helix, 'E' for sheet, and 'C' for random coil
    def set_state_3(self, state):
        self.state_3 = state
    
    # the convert method is No.5 in ref. Current Bioinformatics, 2020, 15, 90-107
    # which is different from previous version.
    def conv_8_3_state(self):
        dict_8_3_state = {}
        dict_8_3_state['H'] = 'H'
        dict_8_3_state['G'] = 'H'
        dict_8_3_state['I'] = 'H'
        dict_8_3_state['E'] = 'E'
        dict_8_3_state['B'] = 'C'
        dict_8_3_state['T'] = 'C'
        dict_8_3_state['S'] = 'C'
        dict_8_3_state['C'] = 'C'
        dict_8_3_state[' '] = 'C'

        self.state_3 = dict_8_3_state[self.state_8]
    

    def set_cs(self, atom_ID, cs_value):
        self.chemical_shift.set_cs(atom_ID, cs_value)
    
    def set_acc(self, acc):
        self.acc = acc
    
    def set_phi(self, phi):
        self.phi = phi
    
    def set_psi(self, psi):
        self.psi = psi
    
    def disp(self):
        print('{}\t{}\t{}\t{}\t{}\t{}\t'.format(self.name, self.state_8, self.state_3, self.acc, self.phi, self.psi), end='')
        self.chemical_shift.disp()
    
    def as_blank(self):
        self.name = 'B'
        self.state_8 = 'S'
        self.state_3 = 'C'
        self.chemical_shift = aa_chem_shift_v2()
        self.acc = 124    # like G in random coil
        self.phi = -150   # like G in random coil
        self.psi = -100   # like G in random coil

        return(self)
        



# ACC, Torsion angles, are added in the v2 version of protein class


class peptide:
    def __init__(self):
        self.pdbid = ''
        self.bmrbid = ''
        self.length_aa_seq = 0
        self.aa_seq = []
        self.amino_acid_seq = []
        self.prob_H_5aa = 0.333         # the default probability of 5-aa peptide in helix
        self.prob_E_5aa = 0.333         # the default probability of 5-aa peptide in sheet
        self.prob_C_5aa = 0.333         # the default probability of 5-aa peptide in random coil
        self.prob_H_3aa = 0.333         # the default probability of 3-aa peptide in helix
        self.prob_E_3aa = 0.333         # the default probability of 3-aa peptide in sheet
        self.prob_C_3aa = 0.333         # the default probability of 3-aa peptide in coil
        self.prob_H_1aa = 0.333
        self.prob_E_1aa = 0.333
        self.prob_C_1aa = 0.333


    
    # protein is a protein_v2 instance here.
    def from_protein(self, protein, idx_start, idx_end):
        self.pdbid = protein.pdbid
        self.bmrbid = protein.bmrbid
        self.aa_seq = protein.aa_seq[idx_start:idx_end]
        self.amino_acid_seq = protein.amino_acid_seq[idx_start:idx_end]
        self.length_aa_seq = len(self.aa_seq)

        self.aa_seq_str = ''
        
        for aa in self.aa_seq:
            self.aa_seq_str += aa
        
        return(self)
        
    def disp(self):
        for aa in self.amino_acid_seq:
            aa.disp()


    # dict_state_prob need be read from state_prob data file
    # for now, dict_state_prob contains the state probability of the central a.a. in a quintuplet (5-aa-length peptide)
    def update_state_prob(self, dict_state_prob, len_piece=5):
        if self.length_aa_seq < 5:
            return(self)
        if self.length_aa_seq % 2 == 0:
            return(self)

        idx_central = self.length_aa_seq // 2
        len_half_piece = len_piece // 2          # 5:2, 3:1, 1:0

        seq_str = ''
        for i in range(idx_central - len_half_piece, idx_central + len_half_piece + 1):
            seq_str += str(self.aa_seq[i])
        
        if seq_str not in dict_state_prob:
            return(self)
        else:
            if len_piece == 5:
                self.prob_H_5aa = dict_state_prob[seq_str][0]
                self.prob_E_5aa = dict_state_prob[seq_str][1]
                self.prob_C_5aa = dict_state_prob[seq_str][2]
            elif len_piece == 3:
                self.prob_H_3aa = dict_state_prob[seq_str][0]
                self.prob_E_3aa = dict_state_prob[seq_str][1]
                self.prob_C_3aa = dict_state_prob[seq_str][2]
            elif len_piece == 1:
                self.prob_H_1aa = dict_state_prob[seq_str][0]
                self.prob_E_1aa = dict_state_prob[seq_str][1]
                self.prob_C_1aa = dict_state_prob[seq_str][2]
            else:
                return(self)

        return(self)

    
    # return the chemical shift values of the central amino acid in this peptide
    def cs_central_aa(self):
        idx_central = self.length_aa_seq // 2
        cs_H = self.amino_acid_seq[idx_central].chemical_shift.cs_H
        cs_HA = self.amino_acid_seq[idx_central].chemical_shift.cs_HA
        cs_N = self.amino_acid_seq[idx_central].chemical_shift.cs_N
        cs_C = self.amino_acid_seq[idx_central].chemical_shift.cs_C
        cs_CA = self.amino_acid_seq[idx_central].chemical_shift.cs_CA
        cs_CB = self.amino_acid_seq[idx_central].chemical_shift.cs_CB

        return([cs_H, cs_HA, cs_N, cs_C, cs_CA, cs_CB])

    # output: (mat_peptide_feature, prob_state, state)
    # state is the integer coding the 2nd structure of the central a.a.
    # prob_state is the probability of the central a.a. in each 2nd structure by statistical analysis
    # 
    # the format of mat_peptide_feature
    # each row is a amino acid
    # column 0: amino acid name coded from 0 to 21
    # column 1-7: physichemical properties
    # column 8-13: chemical shifts
    # column 14-16: acc, phi and psi angles

    def to_numpy_tuple(self):
        
        mat_peptide_feature = np.zeros( (self.length_aa_seq, 17), dtype=np.float32 )

        for i in range(self.length_aa_seq):
            mat_peptide_feature[i, 0] = aa_code(self.aa_seq[i])
            mat_peptide_feature[i, 1:8] = phys_chem_code(self.aa_seq[i])
            mat_peptide_feature[i, 8] = self.amino_acid_seq[i].chemical_shift.cs_H
            mat_peptide_feature[i, 9] = self.amino_acid_seq[i].chemical_shift.cs_HA
            mat_peptide_feature[i, 10] = self.amino_acid_seq[i].chemical_shift.cs_N
            mat_peptide_feature[i, 11] = self.amino_acid_seq[i].chemical_shift.cs_C
            mat_peptide_feature[i, 12] = self.amino_acid_seq[i].chemical_shift.cs_CA
            mat_peptide_feature[i, 13] = self.amino_acid_seq[i].chemical_shift.cs_CB
            mat_peptide_feature[i, 14] = self.amino_acid_seq[i].acc
            mat_peptide_feature[i, 15] = self.amino_acid_seq[i].phi
            mat_peptide_feature[i, 16] = self.amino_acid_seq[i].psi
            # mat_peptide_feature[i, 17] = second_structure_code( self.amino_acid_seq[i].state_3 )

        prob_state = np.array( [self.prob_H_5aa, self.prob_E_5aa, self.prob_C_5aa, self.prob_H_3aa, self.prob_E_3aa, self.prob_C_3aa, self.prob_H_1aa, self.prob_E_1aa, self.prob_C_1aa], dtype=np.float32 )
        
        idx_central = self.length_aa_seq // 2
        state = second_structure_code( self.amino_acid_seq[idx_central].state_3 )
        state = np.array(state, dtype=np.int32)

        return(mat_peptide_feature, prob_state, state)





class protein_v2:
    # pdbid = ''
    # bmrbid = ''
    # length_aa_seq = 0
    # aa_seq = []
    # second_structure_seq = []
    # chem_shift_seq = []
    # acc_seq = []
    # phi_seq = []
    # psi_seq = []
    def __init__(self, pdbid='', bmrbid=''):
        self.pdbid = pdbid
        self.bmrbid = bmrbid
        self.length_aa_seq = 0
        self.aa_seq = []               # list of amino acid name
        self.amino_acid_seq = []       # list of amino acid class instance

    
    # read amino acid sequence, and 2nd structure sequence from dssp file
    # modify aa_seq, second_structure_seq , length_aa_seq, according the read information
    # each element in chem_shift_seq is all assigned chem shifts of ONE amino acid
    def read_seq(self, dir_dssp):
        file_name = dir_dssp + self.pdbid + '.dssp'
        seq_read = read_dssp_v2(file_name)
        
        if seq_read != None:
            (aa_seq, second_structure_seq, acc_seq, phi_seq, psi_seq) = seq_read
            if 'X' in aa_seq or '!' in aa_seq or len(aa_seq) < 25:      # if there is unknown a.a. or break point in sequece, or sequece is shorter than 25, drop this protein
                return(None)
            aa_seq = fix_dssp_abbr(aa_seq)
            self.aa_seq = aa_seq

            for i in range(len(aa_seq)):
                self.amino_acid_seq.append( amino_acid( aa_seq[i] ) )
                self.amino_acid_seq[i].set_state_8( second_structure_seq[i] )
                self.amino_acid_seq[i].conv_8_3_state()
                self.amino_acid_seq[i].set_acc( acc_seq[i] )
                self.amino_acid_seq[i].set_phi( phi_seq[i] )
                self.amino_acid_seq[i].set_psi( psi_seq[i] )

            self.length_aa_seq = len(self.aa_seq)
            return(1)
        else:
            return(None)
    
    def disp(self):
        for aa in self.amino_acid_seq:
            aa.disp()
        
    
    # read assigned chemical shifts from cs file
    # modify chem_shift_seq according to the read information
    # the unassigned chemical shift are set as -100 ppm by default
    # only read the chemical shift file of proteins who has ONLY ONE bmrb_id!

    def read_chem_shift_v2(self, dir_cs):
        cs_file_list = find_cs_file_v2(dir_cs, self.bmrbid)
        num_cs_file = len(cs_file_list)
        if num_cs_file != 1:
            return(None)
        else:
            name = cs_file_list[0]
            file_name = dir_cs + name
            temp = read_cs(file_name)
            if temp == None:
                return(None)
            
            (aa_seq, cs_dict_seq) = temp
            aa_seq = [abbr_reduce(i) for i in aa_seq]
            
            # if the length of aa sequence is shorter than 1/2 of the whole sequence, ignore this chem shift file
            
            if len(aa_seq) < 0.5 * self.length_aa_seq:
                return(None)


            # compare the aa seq read from CS file, with that from DSSP file
            # find the position of matched blocks
            # write the chem shift of these matched blocks to the instance of class "protein"
            # and marked those aa which has been writen chem shifts 
            s = SequenceMatcher(None, aa_seq, self.aa_seq)
            seq_blocks = s.get_matching_blocks()
            for block in seq_blocks:
                (begin_seq_bmrb, begin_seq_dssp, size) = block
                if size != 0:
                    # print(block)
                    # print( aa_seq[ begin_seq_bmrb:(begin_seq_bmrb+size) ] )
                    # print( self.aa_seq[ begin_seq_dssp:(begin_seq_dssp+size) ] )
                    for k in range(size):
                        for atom_ID, cs_value in cs_dict_seq[begin_seq_bmrb+k].items():
                            self.amino_acid_seq[begin_seq_dssp+k].set_cs(atom_ID, cs_value)

            return(1)


    # read NMRSTAR file to fulfill protein instance
    def read_star(self, star_file_name):
        
        # if there are more than one entity in the sample, abort the progress
        flag_single_entity = tell_single_entity_from_star(star_file_name)
        if flag_single_entity == False:
            print('There are more than ONE entity in the sample, while we demand ONE')
            return(None)
        
        # if there are more than one cs_set in the sample, or no cs_set
        # abort the progress
        flag_single_cs_set = tell_single_cs_set_from_star(star_file_name)
        if flag_single_cs_set == False:
            print('We need only ONE chemical shift assignment set')
            return(None)
        
        # read protein seq from tag Entity.Polymer_seq_one_letter_code in NMRSTAR
        aa_seq_1 = read_seq_from_star(star_file_name)
        if aa_seq_1 != None:
            self.aa_seq = aa_seq_1
            self.length_aa_seq = len(aa_seq_1)
        
        # read cs_set from NMRSTAR into pandas framework
        pd_cs_set = read_cs_into_pd_from_star(star_file_name)
        (aa_seq_2, cs_dict_seq) = read_cs_from_pd(pd_cs_set)

        # if it's fail to read protein seq from tag Entity.Polymer_seq_one_letter_code
        # relapce it with aa_seq of cs_set
        if len(self.aa_seq) == 0:
            self.aa_seq = aa_seq_2
            self.length_aa_seq = len(aa_seq_2)

        
        # initiate amino_acid_seq list from aa_seq
        # there are no 2nd structrure, acc, phi, or psi information in NMRSTAR, so we pick default value
        for i in range(self.length_aa_seq):
            self.amino_acid_seq.append( amino_acid( self.aa_seq[i] ) )  
            self.amino_acid_seq[i].set_state_8( 'S' )   
            self.amino_acid_seq[i].set_state_3( 'C' )
            self.amino_acid_seq[i].set_acc( 124 )
            self.amino_acid_seq[i].set_phi( -150 )
            self.amino_acid_seq[i].set_psi( -100 )
        
        # compare the aa seq read from CS_set, with that from tag Entity.Polymer_seq_one_letter_code
        # find the position of matched blocks
        # write the chem shift of these matched blocks to the instance of class "protein"
        # and marked those aa which has been writen chem shifts 
        s = SequenceMatcher(None, aa_seq_2, self.aa_seq)
        seq_blocks = s.get_matching_blocks()
        for block in seq_blocks:
            (begin_seq_bmrb, begin_seq_dssp, size) = block
            if size != 0:
                # print(block)
                # print( aa_seq[ begin_seq_bmrb:(begin_seq_bmrb+size) ] )
                # print( self.aa_seq[ begin_seq_dssp:(begin_seq_dssp+size) ] )
                for k in range(size):
                    for atom_ID, cs_value in cs_dict_seq[begin_seq_bmrb+k].items():
                        self.amino_acid_seq[begin_seq_dssp+k].set_cs(atom_ID, cs_value)
        
        return(1)


    # the name of a.a. be padded is 'B'
    # the chemical shift of padded a.a. are all -100
    def padding(self, size_padding):
        self.length_aa_seq += 2 * int(size_padding)
        self.aa_seq = size_padding * ['B'] + self.aa_seq + size_padding * ['B']
        
        self.amino_acid_seq = [amino_acid('B').as_blank() for i in range(size_padding)] + self.amino_acid_seq + [amino_acid('B').as_blank() for i in range(size_padding)]
       
    # slice protein sequence into small pieces
    def slicing(self, size_slice):
        list_peptide = []
        for idx_start in range(self.length_aa_seq - size_slice + 1):
            idx_end = idx_start + size_slice
            list_peptide.append(peptide().from_protein(self, idx_start, idx_end))
        
        return(list_peptide)
    
    # return the propotion of aa whose assignment is higher than 3 
    def assign_status(self):
        list_aa_assigned_num = []

        for aa in self.amino_acid_seq:
            num_assigned = 0
            if aa.chemical_shift.cs_H > -50:
                num_assigned += 1
            if aa.chemical_shift.cs_HA > -50:
                num_assigned += 1
            if aa.chemical_shift.cs_N > -50:
                num_assigned += 1
            if aa.chemical_shift.cs_C > -50:
                num_assigned += 1
            if aa.chemical_shift.cs_CA > -50:
                num_assigned += 1
            if aa.chemical_shift.cs_CB > -50:
                num_assigned += 1
            list_aa_assigned_num.append(num_assigned)
        
        proportion_3 = len( [i for i in list_aa_assigned_num if i >= 3] ) / len(list_aa_assigned_num)

        return(proportion_3)


    def unpadding(self, size_padding):
        self.length_aa_seq -= 2 * int(size_padding)
        self.aa_seq = self.aa_seq[size_padding : (-1)*size_padding]
        self.amino_acid_seq = self.amino_acid_seq[size_padding : (-1)*size_padding]


    # only for the peptide list which is generated by the above slicing method
    def from_peptide_list(self, list_peptide):
        self.pdbid = list_peptide[0].pdbid
        self.bmrbid = list_peptide[0].bmrbid
        self.aa_seq = []
        self.amino_acid_seq = []

        idx_central = len(list_peptide[0].aa_seq) // 2

        for i in range(len(list_peptide)):
            self.aa_seq.append( list_peptide[i].aa_seq[idx_central] )
            self.amino_acid_seq.append( list_peptide[i].amino_acid_seq[idx_central] )
        
        self.length_aa_seq = len(self.aa_seq)
        

    # one protein record is a numpy 2D array
    # each row is an amino acid
    # there are 18 (1 + 7 + 6 + 3 + 1) elements in each row. 
    # the 0th one is amino acid name coded by integer range from 0 to 21 (20 natural a.a., one unknown a.a., and one blank a.a.)
    # the 1 to 7th columns are physicochemical properties 
    # the 8 to 13rd columns are chemical shift of 6 backbone nuclear spin (H, HA, N, C, CA, CB)
    # the 14th column is acc
    # the 15th column is phi angle
    # the 16th column is psi angle
    # the l7th column is the second structure, 0 for H, 1 for E, and 2 for C
    def to_numpy(self):

        mat_protein = np.zeros( (self.length_aa_seq, 18), dtype=np.float32 )

        for i in range(self.length_aa_seq):
            mat_protein[i, 0] = aa_code(self.aa_seq[i])
            mat_protein[i, 1:8] = phys_chem_code(self.aa_seq[i])
            mat_protein[i, 8] = self.amino_acid_seq[i].chemical_shift.cs_H
            mat_protein[i, 9] = self.amino_acid_seq[i].chemical_shift.cs_HA
            mat_protein[i, 10] = self.amino_acid_seq[i].chemical_shift.cs_N
            mat_protein[i, 11] = self.amino_acid_seq[i].chemical_shift.cs_C
            mat_protein[i, 12] = self.amino_acid_seq[i].chemical_shift.cs_CA
            mat_protein[i, 13] = self.amino_acid_seq[i].chemical_shift.cs_CB
            mat_protein[i, 14] = self.amino_acid_seq[i].acc
            mat_protein[i, 15] = self.amino_acid_seq[i].phi
            mat_protein[i, 16] = self.amino_acid_seq[i].psi
            mat_protein[i, 17] = second_structure_code( self.amino_acid_seq[i].state_3 )

        return(mat_protein)




# ###############################################################
####            main function for test
#################################################################

if __name__ == "__main__":
    


    dir_dssp = '../01_data_preparation_for_protein_structure_pred/myDSSP/'
    dir_cs = '../01_data_preparation_for_protein_structure_pred/ChemShift/'
    
    file_state_prob_5aa = 'quintuplet_state_prob.csv'
    dict_state_prob_5aa = read_state_prob_file(file_state_prob_5aa)

    file_state_prob_3aa = 'triplet_state_prob.csv'
    dict_state_prob_3aa = read_state_prob_file(file_state_prob_3aa)

    file_state_prob_1aa = 'singlet_state_prob.csv'
    dict_state_prob_1aa = read_state_prob_file(file_state_prob_1aa)

    # print(len(dict_state_prob))



    # aa_1 = amino_acid('B')
    # aa_1.as_blank()
    # aa_1.disp()


    p_1a5j = protein_v2("1a5j", "5517")
    p_1a5j.read_seq(dir_dssp)
    p_1a5j.read_chem_shift_v2(dir_cs)
    # p_1a5j.disp()

    # mat_1a5j = p_1a5j.to_numpy()
    # print(mat_1a5j)

    p_1a5j.padding(2)
    # print(p_1a5j.length_aa_seq)
    # p_1a5j.disp()
    # print("\n")

    list_peptide = p_1a5j.slicing(5)

    for peptide in list_peptide[:3]:
        peptide.update_state_prob(dict_state_prob_5aa, len_piece=5)
        peptide.update_state_prob(dict_state_prob_3aa, len_piece=3)
        peptide.update_state_prob(dict_state_prob_1aa, len_piece=1)
    
        print(peptide.aa_seq_str)
        mat_peptide, prob_state, state = peptide.to_numpy_tuple()
        # print(mat_peptide)
        # print(prob_state)
        peptide.disp()
        print(state)

        # print(peptide.cs_central_aa())
    #     print(peptide.prob_H, peptide.prob_E, peptide.prob_C)

    # iter_peptide = iter(list_peptide)

    # for _ in range(1):
    #     peptide = next(iter_peptide)

    # peptide.disp()

    # p_new = protein_v2()
    # p_new.from_peptide_list(list_peptide)
    # print(p_new.length_aa_seq)
    # p_new.disp()

    # print(len(list_peptide))
    # for peptide in list_peptide:
    #     peptide.disp()
    #     print('\n')

    # p_1a5j.unpadding(2)
    # print(p_1a5j.length_aa_seq)
    # p_1a5j.disp()

    # print('{:3.3f}'.format(p_1a5j.assign_status()))









