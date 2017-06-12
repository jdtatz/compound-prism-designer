from compoundprism import *
import sys
from json import load

if len(sys.argv) != 3:
    print("usage compoundprism [run option] [run.json]")
else:
    with open(sys.argv[2], 'r') as f:
        settings = load(f)
    if sys.argv[1] == 'design':
        design(**settings)
    elif sys.argv[1] == 'post':
        post_process(**settings)
    elif sys.argv[1] == 'hyper':
        hyper(**settings)
    else:
        print("invalid run option")
