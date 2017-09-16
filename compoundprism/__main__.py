from compoundprism.designer import design
import sys
from json import load

if len(sys.argv) != 2:
    print("usage compoundprism [run.json]")
else:
    with open(sys.argv[1], 'r') as f:
        settings = load(f)
    design(**settings)

