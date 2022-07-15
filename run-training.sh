# This file was created by IDLab-MEDIA, Ghent University - imec, in collaboration with GRIP-UNINA

#!/bin/bash
echo "START PYTHON SCRIPT TRAINING"
cd /project_ghent/comprint
current_dir=$(pwd)
python3 /project_ghent/public/comprint/code/train_network.py $current_dir $current_dir

