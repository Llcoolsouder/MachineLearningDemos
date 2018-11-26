# Decision Trees

This folder contains classifiers that use the *decision tree* algorithm and variations of this algorithm.  
All decision tree models are implemented using Python's `sklearn` package.


## Dota2 Predictor
The data set a used to test the decision tree algorithm was the Dota2 dataset from <a href="http://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results#"></a>.
Essentially, the classifier would predict whether a team won or lost a game based on team composition, enemy team composition, game mode, and game type.  

The results of the decision tree classifiers are just *barely* better than a coin toss, so I would chalk this up to an overall failure. 

|  Architecture | Number of Trees  | Test Performance  |
|:---|:---:|:---:|
| Decision Tree  | 1 | 51.8% |
| Random Forest | 20 | 55.6% |
| Random Forest  | 100 | 58.0% |

This could be due to the fact that this algorithm is not suited to this data type or there just may not be sufficient data.
For instance, the outcome of the game definitely doesn't depend solely on the team compositions and game modes.  For more accurate predictions, one might include the skill rating of each player in the data.  
It is also possible that this architecure is not well suited to a sparse dataset with many features and only two classes.
