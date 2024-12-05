conda create -n test_2 -y
conda activate test_2
conda install conda-forge::albumentations -y
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install numpy -y
conda install transformers -y
conda install scikit-learn -y
conda install nltk -y
conda install einops -y
conda install huggingface_hub -y
conda install pandas -y
