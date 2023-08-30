# Predictive Uncertainty-based Bias  Mitigation in Ranking

This repository contains the code used for the experiments in 
"Predictive Uncertainty-based Bias  Mitigation in Ranking", 
which will be published at CIKM 2023.

## Citation
If you use this code to produce results for your scientific publication, or if you share a copy or fork, 
please refer to our CIKM 2023 paper:

```
@inproceedings{heuss-2023-predictive,
author = {Heuss, Maria and Cohen, Daniel and Mansoury, Masoud and de Rijke, Maarten and Eickhoff, Carsten},
booktitle = {CIKM 2023: 32nd ACM International Conference on Information and Knowledge Management},
publisher = {ACM},
title = {Predictive Uncertainty-based Bias Mitigation in Ranking},
year = {2023}}
```

## Licence
This repository is published under the terms of the GNU General Public License version 3. 
For more information, see the file LICENSE.

```
Predictive Uncertainty-based Bias  Mitigation in Ranking 
Copyright (C) 2023 Maria Heuss

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
```

## Usage
Packages used in this repository can be found in requirements.txt 

To run the experiments that are recorded in the paper create a folder called 'results'
and run 'run_msmarco.py'.
By choosing the variable 'experiment' in ['table', 'ablation', 'trade-off-curve']
the results of the table, ablation-study resp. trade-off plot can be generated. 
This will create new files in the '\results\' folder one with the full candidate
DataFrame containing a column with the new ranks for each method and one with the 
calculated metrics for each query and each method. 


## Credits 
This project uses parts of the code of
- https://github.com/MilkaLichtblau/BA_Laura, 
  and 
- https://networkx.org, which is published under the 3-clause BSD licence 

for the implementation of the convex optimization baseline. 
