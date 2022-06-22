'''
Author: HelinXu xuhelin1911@gmail.com
Date: 2022-06-16 01:23:00
LastEditTime: 2022-06-23 03:35:46
Description: Train Val Split
'''
import os
import sys
from glob import glob
from misc import DATA_ROOT

if os.path.isdir(os.path.join(DATA_ROOT, 'val')):
    print('Val folder exists!')
    sys.exit(0)
else:
    os.mkdir(os.path.join(DATA_ROOT, 'val'))

for i, file in enumerate(glob(os.path.join(DATA_ROOT, 'train/*'))):
    print(file)
    if i % 10 == 0:
        os.system('mv {} {}'.format(file, os.path.join(DATA_ROOT, 'val/')))