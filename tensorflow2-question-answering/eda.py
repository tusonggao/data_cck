from __future__ import print_function, division, with_statement
import os
import sys
import time
import json
import numpy as np
import pandas as pd

with open('./atad/simplified-nq-test.jsonl') as file_r:
    for line in file_r:
        json_line = json.load(line.strip())
        print('json_line is ', json_line)

print('prog ends here')

