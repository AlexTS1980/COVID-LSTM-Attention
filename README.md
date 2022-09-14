### Update from 14/09/22: Published in IEEE Intelligence Systems, May-June 2022, volume 37, pages 54-64

### Update from 05/12/21: To appear in IEEE Intelligent Systems

Citation of preprint on medRxiv:
```
@article {Ter-Sarkisov2021.02.16.21251754,
	author = {Ter-Sarkisov, Aram},
	title = {One Shot Model For COVID-19 Classification and Lesions Segmentation In Chest CT Scans Using LSTM With Attention Mechanism},
	year = {2021},
	doi = {10.1101/2021.02.16.21251754},
	publisher = {Cold Spring Harbor Laboratory Press},
	journal = {medRxiv}
}
```
Citation of journal publication:
```
@article{9669072,
  author={Ter-Sarkisov, Aram},
  journal={IEEE Intelligent Systems}, 
  title={One Shot Model for COVID-19 Classification and Lesions Segmentation in Chest CT Scans Using Long Short-Term Memory Network With Attention Mechanism}, 
  year={2022},
  volume={37},
  number={3},
  pages={54-64},
  doi={10.1109/MIS.2021.3135474}}
```
# One Shot Model Using LSTM with Attention 

## Segmentation And Classification Results From The Paper:

### Segmentation Results (CNCB-NCOV Segmentation Dataset, (http://ncov-ai.big.ac.cn)

|  \# Affinities	| AP@0.5 	| AP@0.75 	| mAP@[0.5:0.95:0.05] 	| 
|:-:	|:-:	|:-:	|:-:|
|  **One Shot +LSTM+Attention**	| **0.605** 	| **0.497** 	| **0.470** 	| 
| Mask R-CNN |  0.502| 0.419|0.387|

### Classification Results (CNCB-NCOV Classification Dataset, (http://ncov-ai.big.ac.cn)

|  \# Model	| COVID-19 	| CP 	| Normal 	| F1 score|
|:-:	|:-:	|:-:	|:-:|:-:|
| **One Shot + LSTM+Attention**	|**95.74%**	|**98.13%**|**99.27%** 	|**98.15%**|
| ResNet50 | 91.04% |97.64%| 98.97%|96.88%|
| ResNeXt50 | 91.94% |88.45%| 84.30%|87.31%|
| ResNeXt101 | 91.58% |92.13%| 94.02%|92.86%|
| DenseNet121 | 92.64% |96.16%| 98.98%|96.15%|

### Classification Results (iCTCF-CT Classification Dataset, (http://ictcf.biocuckoo.cn)

|  \# Model	| COVID-19 	| Normal 	| F1 score|
|:-:	|:-:	|:-:	|:-:|
|**One Shot + LSTM + Attention**	| **97.73%**	|**98.68%**	|**98.41%** |
|VGG16(Ning et al, 2020) | 97.00%	|85.47%|-|

### The Model:
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-LSTM-Attention/blob/master/Images/im1.png" width="800" height="250" align="center"/>
</p>

### Attention Layer:
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-LSTM-Attention/blob/master/Images/im2.png" width="800" height="250" align="center"/>
</p>

### LSTM + Attention:
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-LSTM-Attention/blob/master/Images/im3.png" width="700" height="450" align="center"/>
</p>

Due to the size of the backbone (ResNext101+FPN), we provide the second-best model, with ResNext50+FPN backbone. 

To train the model, simply run 
```
python train.py
```
on the CNCB-NCOV data. You need both segmentation and classification splits, see https://github.com/AlexTS1980/COVID-Single-Shot-Model for details. 
To evaluate the provided model, change the path in `eval.py` before running: 
```
python eval.py
```
