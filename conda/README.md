# Conda Environment

## Installation

> **_NOTE:_**  The following instructions are for installing the conda package manager on a Linux system. For other systems, please refer to the [official conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Download and install the `conda` matching your OS: <https://docs.conda.io/en/latest/miniconda.html>.

Run the following command:

```bash
# Assuming you're in the '~'(/home/<you>) directory

# Download the installer and install it
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh

# Activate conda
$ cd ~/miniconda3/condabin/
$ ./conda init

# Restart the terminal and 'cd' into the project directory
(base) $ cd ~/<path_to_project>

# Create a new environment and install the required packages
(bash) $ cd conda/
(base) $ conda env create -f conda_env.yaml
(base) $ conda activate macrograd
(macrograd) $ python -m pip install -r ../requirements.txt
```

## Visualization

In this project, we use `graphviz` to visualize the computational graph. In addition to install python package `graphviz`, we need to install the `graphviz` package.

```bash
sudo apt install graphviz
```
