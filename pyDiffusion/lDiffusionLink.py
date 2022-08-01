""" 
Dynamic loader for libDiffusion.so
"""
import sys
import os 

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(path)

import libDiffusion