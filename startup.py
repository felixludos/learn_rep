print('importing src')
import sys, os
# sys.path.insert(1,os.getcwd())
print('__file__', __file__)
print(os.getcwd())
print(sys.path)
import src
print(src.__file__)
print('finished importing src')

