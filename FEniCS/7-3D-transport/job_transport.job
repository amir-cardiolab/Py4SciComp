#!/bin/bash                                                                                                                                                                                                                                                                                        
#SBATCH --job-name=transport
#SBATCH --time=40:30:00     # 20 min, shorter time, quicker start, max run time    
#SBATCH --chdir=/home/sci/amir.arzani/Python_tutorials/Fenics/NS_steady/      # your work directory                                                                                                                                                                                                           
###SBATCH --mem=48000                     # memory                                                                                                                                                                                                                                            
#SBATCH --nodes=1                                                                                                                                                                                                                                                                                  
#SBATCH --ntasks=30                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
###SBATCH --partition=arzani 
###SBATCH -C sl
###SBATCH --gres=gpu:tesla:1                                                                                                                                                                                                                                                                      


source ~/anaconda3/etc/profile.d/conda.sh
conda activate fenics2018							   
#time mpirun  python /home/sci/amir.arzani/porous/ML/Stokes_generatedata_stenosis.py > k1.out


#conda activate fenicsproject	
time mpirun python adv-diff-Dif2-CN.py > log.out


