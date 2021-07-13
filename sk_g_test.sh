#!/bin/bash
set -e
source activate tf
cd /share/pj/rb
for ii in 4 8; do
 python main.py test --n_dl 16 --dsid skin4 --mid res50 --midtf skin4_res50_g --batch_attack 64 --testidtf skin4_res18_b_eval_FGSM_e$ii
 #python main.py test --n_dl 16 --dsid skin4 --mid res50 --midtf skin4_res50_g --batch_attack 64 --testidtf skin4_res18_b_eval_IFGSM_ei$ii,5
done
