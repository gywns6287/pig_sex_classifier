# Sex estimation from pig carcasses

## Object

The sex of pig carcasses is one of the important factors that have an influence on pork price.  Currently,  trained human inspect all pig carcasses one by one, taking time and effort. Therefore, a more efficient method to determining the sex of pig carcasses should be developed.

## Model & Code

our models were implemented by **tensorflow 2.3** and **keras**

#### Model summary
1. This model was based on ResNet50.
2. Class Activation Map was printed together with output results.
3. Model code can be found in the `CAM_LHJ.py`.

#### Example with training
```
python solution.py --test data\\test --train data\\train --out .
```
#### Example without training
```
python solution.py --test data\\test --out .
```
The pre-trained weight must exist as `weight.h5` in the path where `solution.py` is located.

## Datasets & pre-trained weight

The `data` folder contains only simple sample images and does not contain the data used for model training.  All the data-set used in this study was provided by Artificial Intelligence Convergence Research Center(Chungnam National University)). Request for academic urposes can be made to gywns6298@nvaer.com.

pre-trained weight can be downloaded at https://drive.google.com/drive/folders/1jcsroiExir9e4PKU6kFVmHBAZLdaAMx5?usp=sharing


## Model output

1. `pred.sol`: estimation results for test image set.
2. `summary.txt`: summary of test results (accuracy, F1-score).
3. **CAM**
	male
  !(C:\Users\user\Desktop\my_scripts\for_git\pig_sex_classifier\results\B-2020.07.29-09-41-01-0000430 C1.tif_male_male_1.0.png)
	female
  ![female](C:/Users/user/Desktop/my_scripts/for_git/pig_sex_classifier/results/B-2020.07.20-11-47-36-0000702 C1.tif_female_female_1.0.png)
	boar
  ![B-2020 03 17-09-32-48-0000418 C1 tif_boar_boar_1 0](https://user-images.githubusercontent.com/71325306/94219434-cd349380-ff21-11ea-9f99-e1b91adda17b.png)
