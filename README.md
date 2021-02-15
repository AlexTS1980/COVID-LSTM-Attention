# One Shot Model Using LSTM with Attention 

### Segmentation Results (CNCB-NCOV Segmentation Dataset, (http://ncov-ai.big.ac.cn)

|  \# Affinities	| AP@0.5 	| AP@0.75 	| mAP@[0.5:0.95:0.05] 	| 
|:-:	|:-:	|:-:	|:-:|
|  One Shot +LSTM+Attention	| 0.587 	| 0.465 	| 0.442 	| 
| One Shot + LSTM | **0.612** 	| **0.485** 	| **0.466** 	|
| Mask R-CNN |  0.502| 0.419|0.387|

### Classification Results (CNCB-NCOV Classification Dataset, (http://ncov-ai.big.ac.cn)

|  \# Model	| COVID-19 	| CP 	| Normal 	| F1 score|
|:-:	|:-:	|:-:	|:-:|:-:|
| One Shot + LSTM+Attention	|**95.35%**	|**98.20%**|99.30% 	|**98.10%**|
| One Shot + LSTM | 93.58% 	|96.28% 	|**99.68%** |97.24% 	|
| ResNet50 | 91.04% |97.64%| 98.97%|96.88%|
| ResNeXt50 | 91.94% |88.45%| 84.30%|87.31%|
| DenseNet121 | 92.64% |96.16%| 98.98%|96.15%|

### Classification Results (iCTCF-CT Classification Dataset, (http://ictcf.biocuckoo.cn)

|  \# Model	| COVID-19 	| Normal 	| F1 score|
|:-:	|:-:	|:-:	|:-:|
|  One Shot + LSTM + Attention	| 97.73%	|**98.68%**	|**98.41%** |
| One Shot + LSTM | **99.37%**	|97.36% |97.94%|
|VGG16(Ning et al, 2020) | 97.00%	|85.47%|-|
