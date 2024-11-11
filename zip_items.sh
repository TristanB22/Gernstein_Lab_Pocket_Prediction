#!/bin/bash
#SBATCH --time=5:00:00

zip -r refined-set.zip refined-set

rm -r refined-set
