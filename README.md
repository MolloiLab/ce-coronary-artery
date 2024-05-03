# marimo-imagecas introduction

Testing

# virtual environment setup
### Prerequisite
You should login in to the cluster. \
You can be in any node(server). I recommand you to login to the free CPU server.

### Install Mini-conda
We are using Mini-conda for enviornment.
- run `cd` first to get to the default path.
- then run the following commands.
- **Before running:** if you don't want to install conda to directory `/miniconda`, change the path in the scripts.
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After running the scripts:
- run `~/miniconda3/bin/conda init bash` 
  

Now, you should be able to use `conda` commands.
- run `conda --version` to make sure your conda is ready.
- if you are still not able to use `conda` commands, try `source .bashrc`, which restart/refresh the bash shell.

### Create Env
- `conda create --name imagecas python=3.9` to create a virtual environment with python version 3.9
- `conda activate imagecas` to run the virtual environment
- `conda install pip` to install `pip` package manager for the virtual environment
- you have to check if you are using the correct pip by `pip --version` and see the path of the pip. It should be some path that contains miniconda3 like `.../miniconda3/env/imagecas...`
- if your pip is not having the correct path, contact Liam. (I think you should delete the pyenv settings in .bashrc or .bash_profile Andrew :))
- [unnecessary] you can check if you are using the correct python version by `python --version` and see the path of the pip

### Install Dependencies
- `cd` to the directory you clone this repo. If you didn't clone the repo, clone it first, then open it in terminal.
- run `pip install -r requirements.txt`
- done :)

### Run the project with marimo
- in your environment, do `pip install marimo`
- run `marimo edit [filename] --host 0.0.0.0 --port [num]` to run the marimo file, an example is `marimo edit newTest.py --host 0.0.0.0 --port 1234
- do ssh port forwarding mentioned in `Zulip` group chat

