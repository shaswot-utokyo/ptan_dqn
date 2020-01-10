# USAGE: python seed_gen.py <no_of_seeds> <filename_to_store_seeds>
# Creates a file with seeds separated by newline.

import sys
import numpy as np

NO_OF_SEEDS = int(sys.argv[1]) 
FILENAME = str(sys.argv[2])
np.random.seed(314159)
seed_list = np.random.randint(low=100000, high=999999, size=NO_OF_SEEDS)
np.savetxt(FILENAME, seed_list, fmt="%d") 

