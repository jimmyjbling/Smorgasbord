from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import PandasTools
from datetime import datetime

from rdkit import rdBase

#import sys
#sys.path.append("../
#from chemical_curation.modification_graph import Modification
#from chemical_curation.modification_graph import Modification_Graph

import molvs.normalize
import molvs.fragment
import molvs.tautomer
import molvs.metal

rdBase.DisableLog('rdApp.*')

import os

import math #for rounding

import pandas

import logging
import pathlib

#list of atoms allowed for dragon descriptor calculation
dragon_allowed_atoms = set(["H","B","C","N","O","F","Al","Si","P","S","Cl","Cr",
    "Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Mo","Ag","Cd","In",
    "Sn","Sb","Te","I","Gd","Pt","Au","Hg","Ti","Pb","Bi"])

def curate_mol(mol):

    history = Modification_Graph(mol)

    if mol == None:
        return None, history

    #removal of mixtures
    fragmenter_object = molvs.fragment.LargestFragmentChooser(prefer_organic = True)
    newmol = fragmenter_object.choose(mol)
    if newmol is None:
        history.add_modification(text = "REJECT: Fragment chooser failed")
        history.rejected = True
        return None, history

    if Chem.MolToInchi(newmol) != Chem.MolToInchi(mol):
        history.add_modification(text = "Detected mixture, chose largest fragment")
        mol = newmol

    #removal of inorganics
    if not molvs.fragment.is_organic(mol):
        history.add_modification(text = "REJECT: Molecule is not organic")
        history.rejected = True
        return None, history

    #removal of salts
    remover = SaltRemover.SaltRemover()
    newmol = remover.StripMol(mol, dontRemoveEverything=True) #tartrate is listed as a salt? what do?
    if newmol is None:
        history.add_modification(text = "REJECT: Salt removal failed")
        history.rejected = True
        return None, history

    if Chem.MolToInchi(newmol) != Chem.MolToInchi(mol):
        self.history.add_modification(text = "Detected salts, removed")
        mol = newmol

    #structure normalization
    normalizer = molvs.normalize.Normalizer(normalizations=molvs.normalize.NORMALIZATIONS,
            max_restarts = molvs.normalize.MAX_RESTARTS)
    newmol = normalizer.normalize(mol)
    if newmol is None:
        history.add_modification(text = "REJECT: Normalization failed")
        history.rejected = True
        return None, history

    if Chem.MolToInchi(newmol) != Chem.MolToInchi(mol):
        self.history.add_modification(text = "Normalization(s) applied")
        mol = newmol

    #tautomer selection
    tautomerizer = molvs.tautomer.TautomerCanonicalizer(transforms=molvs.tautomer.TAUTOMER_TRANSFORMS, scores =
            molvs.tautomer.TAUTOMER_SCORES, max_tautomers=molvs.tautomer.MAX_TAUTOMERS)
    newmol = tautomerizer(mol)
    if newmol is None:
        history.add_modification(text = "REJECT: Tautomerization failed")
        history.rejected = True
        return None, history

    if Chem.MolToInchi(newmol) != Chem.MolToInchi(mol):
        history.add_modification(text = "Tautomer(s) canonicalized")
        mol = newmol

    #disconnect metals
    metal_remover = molvs.metal.MetalDisconnector()
    newmol = metal_remover.disconnect(mol)
    if newmol is None:
        history.add_modification(text = "REJECT: Metal removal failed")
        history.rejected = True
        return None, history

    if Chem.MolToInchi(newmol) != Chem.MolToInchi(mol):
        self.history.add_modification(text = "Metal(s) disconnected")
        mol = newmol

    #final check for only valid atoms
    passed_valid, invalid_atom = check_valid_atoms(mol)
    if not passed_valid:
        history.add_modification(text = f"REJECT: '{invalid_atom}' atom not in list of allowed atoms")
        history.rejected = True
        return None, history

    history.add_modification(text = f"Passed validation", mol = mol)
    history.passed = True

    if Chem.MolToSmiles(mol) != history.initial_smiles:
        history.structure_modified = True

    return mol, history

def check_valid_atoms(mol, allowed_list = dragon_allowed_atoms):
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        if s not in allowed_list:
            return False, s
    return True, None


class Modification_Graph:

    def __init__(self, mol, head = None):

        self.head = head
        self.rejected = False
        self.passed = False
        self.structure_modified = False
        try:
            self.initial_smiles = Chem.MolToSmiles(mol)
            self.add_modification(text = f"Initialized [{Chem.MolToSmiles(mol)}]")
        except:
            self.initial_smiles = None
            self.add_modification(text = "REJECT: Mol is None")
            self.rejected = True


    def set_head(self, head):

        self.head = head

    def get_head(self):

        return self.head

    def add_modification(self, text, mol = None):
        if mol:
            text += f" [{Chem.MolToSmiles(mol)}]"

        mod = Modification(text = text)


        if self.head:
            mod.set_parent(self.head)
            self.head.set_child(mod)

        self.head = mod

    def merge(self, other):

        merge_mod = Modification(text = "merge")
        merge_mod.set_parent(self.head)
        merge_mod.set_parent(other.get_head())
        other.get_head().set_child(merge_mod)
        self.head.set_child(merge_mod)
        self.head = merge_mod

    def print(self):

        curr_mod = self.head
        print('\n', curr_mod)
        while curr_mod.parents != None:
            curr_mod = curr_mod.parents[0]
            print(curr_mod)

    #returns True if any modification has `text` as a substring
    def has_modification(self, text):

        nodelist, count = self._depth_first_traversal(node = self.head, nodelist = [], count = 0)
        print(nodelist)
        found = False
        for node in nodelist:
            if type(node) == Modification:
                if text in node.text:
                    found = True
                    break

        return found

    def _depth_first_traversal(self, node, nodelist, count):

        nodelist.append(node)
        count += 1

        if node.parents == None:
            return nodelist, count
        else:
            for parent in node.parents:
                if len(node.parents) > 1:
                    nodelist.append("BRANCH")
                ret_nodelist, ret_count = self._depth_first_traversal(parent, nodelist, count)
                #print("EXISTING: ",  nodelist)
                #print("FROM CALL: ", ret_val)
                nodelist = ret_nodelist
                count = ret_count

            return nodelist, count


    def __contains__(self, s):

        nodelist, count = self._depth_first_traversal(node = self.head, nodelist = [], count = 0)
        for node in nodelist:
            if s in node.text:
                return True

        return False

    def __len__(self):

        if self.head == None:
            return 0
        else:
            '''
            count = 1
            curr_mod = self.head
            while curr_mod.parents != None:
                count += 1
                curr_mod = curr_mod.parents[0]

            return count
            '''
            nodelist, count = self._depth_first_traversal(self.head, [], 0)
            print(nodelist)
            return count

    def __repr__(self):
            nodelist, count = self._depth_first_traversal(self.head, [], 0)
            return "\n".join(reversed([str(node) for node in nodelist]))




class Modification:
    ''' Stores a single modification to a chemical structure or associated data as a simple string.
    Acts as a node in a Modification_Graph
    Can have multiple parents (for merging structures) but only one child
    Examples: "Change =O to -OH", "'Na+' detected as salt and removed"

    '''

    def __init__(self, text, parent = None, child = None):
        self.text = text
        self.timestamp = datetime.now()
        if parent:
            self.parents = [parent]
        else:
            self.parents = None
        if child:
            self.child = child
        else:
            self.child = None

    def set_child(self, child):
        self.child = child

    def set_parent(self, parent, append = True):
        if not self.parents:
            self.parents = []
        if append:
            self.parents.append(parent)
        else:
            self.parents = [parent]

    def __repr__(self):

        return f"{self.timestamp}: {self.text}"

    def __str__(self):

        return self.__repr__()
