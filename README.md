# Estimating Fish Species Weights 
![image](https://user-images.githubusercontent.com/62911364/101485182-6881b600-3920-11eb-8134-eca13f44da32.png)

## Background
Create a machine learning algorithm to accurately estimate the weight of different fish species given multiple measurements. 

## Goals
- Explore the fish [data](https://www.kaggle.com/akdagmelih/multiplelinear-regression-fish-weight-estimation/data) found on kaggle. 
- Follow the appropriate steps of the data science pipeline and deliver a 5 minute [presentation](https://www.canva.com/design/DAEPkmj_Jw8/lNjX0dpNmLdqQPXLfd7SEA/view?utm_content=DAEPkmj_Jw8&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink), a github repository, [project_walkthru.ipynb](https://github.com/george887/estimating_fish_weights/blob/master/project_walkthru.ipynb) notebook and [README.md](https://github.com/george887/estimating_fish_weights/blob/master/README.md).
- Create all the necessary files in analysis. 

## Data Dictionary
| Feature                 | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| species                  | Species of fish                                            |
| weight                   | Weight of fish in lbs                                      |
| vertical_length          | Vertical length of fish in inches                          |
| diagonal_length          | Diagonal length of fish in inches                          |
| cross_length             | Cross length of fish in inches                             |
| height         | Height of fish in inches                                             |
| width                    | Width of fish in inches                                    |
| avg_lengths            | Average of vertical, diagonal and cross lengths              |
| est_area             | Estimated area of a fish. Used the area of an ellipse to estimate fish area |

## Initial Thoughts
Find a way to calculate the area of a fish to help with model. Width will be the missing piece to get a volume estimation. 
### Hypothesis Testing
<details>
  <summary> Click to Expand </summary>
  
> H<sub>0</sub>: There is not linear correlation between the height and weights of fish.

> H<sub>a</sub>: There is a linear correlation between height and weight of fish.
- Pearson Correlation Test
    - P-value less than alpha = .05 so we reject the null hypothesis

 H<sub>0</sub>: There is not linear correlation between the width and weights of fish.

> H<sub>a</sub>: There is a linear correlation between width and weight of fish.
- Pearson Correlation Test
    - P-value less than alpha = .05 so we reject the null hypothesis

 H<sub>0</sub>: There is not linear correlation between the average lengths and weights of fish.

> H<sub>a</sub>: There is a linear correlation between average lengths and weight of fish.
- Pearson Correlation Test
    - P-value less than alpha = .05 so we reject the null hypothesis

 H<sub>0</sub>: There is not linear correlation between the height and weights of fish.

> H<sub>a</sub>: There is a linear correlation between height and weight of fish.
- Pearson Correlation Test
    - P-value less than alpha = .05 so we reject the null hypothesis

</details>

## Project Steps
### Acquire 
Data acquired from kaggle and can be found [here](https://www.kaggle.com/akdagmelih/multiplelinear-regression-fish-weight-estimation/data). Data contains 7 inital columns and 159 observations.

### Prepare
- Data contained no nulls.
- Columns and species names were lower cased. 
- Length columns were renamed to vertical_length, diagonal_length and cross_length and an avg_length column was made with these three measurements. 
- Weight was converted from grams to pounds.
- Lengths were converted from centimeters to inches.
- The area of a fish was estimated using the ellipse area equation ```pi*(avg_lengths /2) * (height /2)```.
- Data split to train, validate and test .
- Data scaled using the ```MinMaxScaler```.
- Functions can be found in the [prepare.py](https://github.com/george887/estimating_fish_weights/blob/master/prepare.py).

### Explore
- Data visualized can be found [here](https://github.com/george887/estimating_fish_weights/blob/master/explore.ipynb).
- Statistical tests performed to see if there was not a linear correlation with features and weight. 

### Model
The data was not normally distributed so the median was used in order to calculate a baseline. The baseline showed a median weight of .66 lbs for the fish in the data set. 
- Linear Regression, LasoLars and Polynomial Regression performed to predict the weight.
- Median absolute error used to determine the middle degree of variation between the model prediction and the actual weight.
- All features used on the first itteration of tests which performed decent.
- Second itteration of tests used ```height```, ```width```, ```avg_lengths```, ```est_area``` and performed better than the model with all the features.
- Validated using the 4 features above with the three regression models.
- 2nd degree Polynomial model performed the best with an average median absolute error of 0.045 lbs across train, validate and test. 

