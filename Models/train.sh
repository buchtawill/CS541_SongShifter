#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64g
#SBATCH -J "CONV_LINEAR_tensorboard"
#SBATCH -p short
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H100
#SBATCH --mail-user=jwbuchta@wpi.edu
#SBATCH --mail-type=BEGIN,FAIL,END

module purge
module load slurm #cuda12.1 #python/3.12.4

now=$(date)
echo "INFO [run.sh] Starting execution on $now"
echo "INFO [run.sh] Training for CONV LINEAR architecture"

#source /home/jwbuchta/CS539_Project/Autoencoder/venv_autoencoder/bin/activate
#which $HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python
# $HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python get_song_y.py
if $HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python train_2.py
then
    echo "INFO  [run.sh] Training successful"
else
    echo "ERROR [run.sh] Training failed"
fi
# tar czf fma_medium.tar.gz ./fma_medium

now=$(date)
echo "INFO [run.sh] Finished execution at $now"
