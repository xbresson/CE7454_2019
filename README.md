# CE7454_2019
Deep learning course CE7454, 2019


<br><br>


### Cloud Machine #1 : Google Colab (Free GPU)

* Follow this Notebook installation :<br>
https://colab.research.google.com/github/xbresson/CE7454_2019/blob/master/codes/installation/installation.ipynb

* Open your Google Drive :<br>
https://www.google.com/drive

* Open in Google Drive Folder 'CE7454_2019' and go to Folder 'CE7454_2019/codes/'<br>
Select the notebook 'file.ipynb' and open it with Google Colab using Control Click + Open With Colaboratory



<br><br>

### Cloud Machine #2 : Binder (No GPU)

* Simply [click here]

[Click here]: https://mybinder.org/v2/gh/xbresson/CE7454_2019/master



<br><br>

### Local Installation

* Follow these instructions (easy steps) :


```sh
   # Conda installation
   curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh # Linux
   curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh # OSX
   chmod +x ~/miniconda.sh
   ./miniconda.sh
   source ~/.bashrc
   #install https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe # Windows

   # Clone GitHub repo
   git clone https://github.com/xbresson/CE7454_2019.git
   cd CE7454_2019

   # Install python libraries
   conda env create -f environment.yml
   source activate deeplearn_course

   # Run the notebooks
   jupyter notebook
   ```




<br><br><br><br><br><br>