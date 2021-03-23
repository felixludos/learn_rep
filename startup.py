#print('importing src')
#import sys, os
# # sys.path.insert(1,os.getcwd())
#
# bad = [i for i, x in enumerate(sys.path) if 'mpi-cluster' in x]
# print(bad)
# for i in bad:
# 	del sys.path[i]
#
#print('__file__', __file__)
#print(os.getcwd())
#print(sys.path)
# from . import src
import src
#print(src.__file__)
#print('finished importing src')

