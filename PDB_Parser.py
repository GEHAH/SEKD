import gzip

import numpy as np
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.PDB import PDBParser, MMCIFParser

def euclidean_distances(x, y, squared=False):
    """
    Compute pairwise (squared) Euclidean distances.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances


class StructureDataParser:
    """
    PDB文件读取
    """

    def __init__(self, path, protein_id, file_type='pdb'):
        """
        init function
        :param path: file path
        :param protein_id: protein id
        :param file_type: Type of the file to be read，the file type can be pdb or mmcif
        """
        if path.endswith('gz'):
            self.path = gzip.open(path, "rt", encoding='UTF-8')
        else:
            self.path = open(path, "rt", encoding='UTF-8')
        self.protein_id = protein_id
        self.file_type = file_type
        if self.file_type == 'pdb':
            self.parser = PDBParser()
        elif self.file_type == 'mmcif':
            self.parser = MMCIFParser()
        else:
            raise ValueError(f"{file_type} is not pdb or mmcif")
        self.structure = self.parser.get_structure(self.protein_id, self.path)
        self.sequence = self.get_sequence()
        self.sequence_len = len(self.sequence)

    def generate_atom_distance_map(self, atom='CA'):
        coords_ = self.generate_residue_coordinate(atom)
        return euclidean_distances(coords_, coords_)

    def get_residues(self):
        return [res for res in self.structure.get_residues()]

    def get_sequence(self):
        return [protein_letters_3to1[res.resname] for res in self.structure.get_residues()]

    def generate_residue_coordinate(self, atom='CA'):#获得CA的三维坐标
        coord_list = [res[atom].coord for res in self.structure.get_residues()]
        coords_ = np.stack(coord_list)
        return coords_

    # def generate_residue_coordinate(self, atoms=['CA','N','C']):
    #     coord_list = []
    #
    #     for res in self.structure.get_residues():
    #         for atom in atoms:
    #             if atom in res:
    #                 coord_list.append(res[atom].coord)
    #                 break  # 如果找到了指定原子，则不再检查其他原子
    #
    #     if coord_list:
    #         coords_ = np.stack(coord_list)
    #         return coords_
    #     else:
    #         return None  # 或者你可以根据需要返回其他值或引发异常

    def get_residue_atoms_coords(self, atoms=None):
        if atoms is None:
            # atoms = ['CA', 'C', 'N']
            atoms = ['CA']
        coords = {atom: self.generate_residue_coordinate(atom).tolist() for atom in atoms}
        # coords = {atom: self.generate_residue_coordinate().tolist() for atom in atomss}
        return coords