# Deep Learning for the plant disease detection

Training and evaluating state-of-the-art deep architectures for plant disease classification task using pyTorch. <br/>
Models are trained on the preprocessed dataset which can be downloaded [here](https://drive.google.com/open?id=0B_voCy5O5sXMTFByemhpZllYREU).<br/>
Dataset is consisted of **38** disease classes from [PlantVillage](https://plantvillage.org/) dataset and **1** background class from Stanford's open dataset of background images - [DAGS](http://dags.stanford.edu/projects/scenedataset.html).
<br/>
**80%** of the dataset is used for training and **20%** for validation.

## Usage:
 1. Train all the models with **train.py** and store the evaluation stats in **stats.csv**:
 `python3 train.py`
 2. Plot the models' results for every archetecture based on the stored stats with **plot.py**:
 `python3 plot.py`
 
 ## Results:
The models on the graph were retrained on final fully connected layers only - **shallow**, for the entire set of parameters - **deep** or from its initialized state - **from scratch**. 

 | Model        | Training type |Training time [~h] | Accuracy      |
| ------------- |:-------------:|:-----------------:|:-------------:|
| AlexNet       | shallow       |    0.87           |   0.9415      |  
| AlexNet       | from scratch  |    1.05           |   0.9578      |  
| AlexNet       | deep          |    1.05           |   0.9924      |
| **DenseNet169**   | **shallow**       |    **1.57**           |   **0.9653**      |    
| **DenseNet169**   | **from scratch**  |    **3.16**           |   **0.9886**      |
| DenseNet169   | deep          |    3.16           |   0.9972      |
| Inception_v3  | shallow       |    3.63           |   0.9153      |
| Inception_v3  | from scratch  |    5.91           |   0.9743      |
| **Inception_v3**| **deep**      |    **5.64**           |   **0.9976**  |
| ResNet34      | shallow       |    1.13           |   0.9475      |
| ResNet34      | from scratch  |    1.88           |   0.9848      |
| ResNet34      | deep          |    1.88           |   0.9967      |
| Squeezenet1_1 | shallow       |    0.85           |   0.9626      |
| Squeezenet1_1 | from scratch  |    1.05           |   0.9249      |
| Squeezenet1_1 | deep          |    2.10           |   0.992       |
| VGG13         | shallow       |    1.49           |   0.9223      |
| VGG13         | from scratch  |    3.55           |   0.9795      |
| VGG13         | deep          |    3.55           |   0.9949      |

**NOTE**: All the others results are stored in [stats.csv](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases/blob/master/Results/stats.csv) 
## Graph
![Results](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases/blob/master/Results/results.png "Results")
