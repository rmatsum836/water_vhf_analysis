import os

import mdtraj as md


trj = md.load('xmolout.xyz', top='top.pdb')

trj.save('try_convert_to_.xtc')
