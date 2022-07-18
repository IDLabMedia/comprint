# This file was created by IDLab-MEDIA, Ghent University - imec, in collaboration with GRIP-UNINA

#!/bin/bash
echo "START PYTHON SCRIPT TRAINING"
current_dir=$(pwd)
python3 ./code/train_network.py $current_dir $current_dir

