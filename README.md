# Pretraining Graph Transformers with Atom-in-a-Molecule Quantum Properties for Improved ADMET Modeling

This repository contains the code to reproduce the results presented in [this paper](). The Graphormer model used here is the one from [chytorch](https://github.com/chython/chytorch). 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

To install the necessary packages run the following command:

```bash
pip install -r requirements.txt
```

## Usage
In order to run trainings and pretrainings use the following command:
```bash
python main.py ./yaml_files/yaml_file_of_choice.yaml
```
where 'yaml_file_of_choice.yaml' is a placeholder for the chosen file.

Curated datasets for pretraining are provided [here](https://zenodo.org/records/13374020), place them in a directory and change paths in the code accordingly. TDC datasets download and handling is done using [PyTDC](https://github.com/mims-harvard/TDC)

For what concerns the analysis they can be run using the provided notebooks.

## License
The code is provided under MIT license, for the curated datasets we refer to the licenses reported in the corresponding links.