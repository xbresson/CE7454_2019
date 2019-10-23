import torch
import math
from rdkit import Chem


class Dictionary(object):
    """
    worddidx is a dictionary
    idx2word is a list
    """


    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2num_occurence = {}
        self.idx2num_occurence = []

    def add_word(self, word):
        if word not in self.word2idx:
            # dictionaries
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            # stats
            self.idx2num_occurence.append(0)
            self.word2num_occurence[word] = 0

        # increase counters    
        self.word2num_occurence[word]+=1
        self.idx2num_occurence[  self.word2idx[word]  ] += 1

    def __len__(self):
        return len(self.idx2word)




def augment_dictionary(atom_dict, bond_dict, list_of_mol ):

    """
    take a lists of rdkit molecules and use it to augment existing atom and bond dictionaries
    """
    for idx,mol in enumerate(list_of_mol):

        for atom in mol.GetAtoms():
            atom_dict.add_word( atom.GetSymbol() )

        for bond in mol.GetBonds():
            bond_dict.add_word( str(bond.GetBondType()) )

        # compute the number of edges of type 'None'
        N=mol.GetNumAtoms()
        if N>2:
            E=N+math.factorial(N)/(math.factorial(2)*math.factorial(N-2)) # including self loop
            num_NONE_bonds = E-mol.GetNumBonds()
            bond_dict.word2num_occurence['NONE']+=num_NONE_bonds
            bond_dict.idx2num_occurence[0]+=num_NONE_bonds




def make_dictionary(list_of_mol_train, list_of_mol_val, list_of_mol_test):

    """
    take three lists of smiles (train, val and test) and build atoms and bond dictionaries
    """
    atom_dict=Dictionary()
    bond_dict=Dictionary()
    bond_dict.add_word('NONE')
    print('train')
    augment_dictionary(atom_dict, bond_dict, list_of_mol_train )
    print('val')
    augment_dictionary(atom_dict, bond_dict, list_of_mol_val )
    print('test')
    augment_dictionary(atom_dict, bond_dict, list_of_mol_test )
    return atom_dict, bond_dict







  

 
        

   