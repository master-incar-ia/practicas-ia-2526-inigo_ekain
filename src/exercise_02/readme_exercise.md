
# Exercise 1: Learn a linear function with PyTorch

## Objective

Estimation of a unknown function by a machine learning model

## Task Formalization

The task is a **regression problem**. We want to learn a mapping $f(x)$ that maps an input scalar $x$ to an output scalar $y$, such that the predicted $\hat{y}$ is as close as possible to the ground truth $y$.

### Task Formalization (Inference)

Given a new, unseen input $x_{new}$, the model must predict the corresponding output $\hat{y}_{new}$.
$$\hat{y}_{new} = f(x_{new},W)$$
Where $W$ represents the learned parameters (weights and biases) of the model.
### Task Formalization (Training)

Write your answer here

## Evaluation metrics

Write your answer here

## Data Considerations

### Dataset description

We are using a dataset of recolected function output with noise, composed of $100$ samples.

### Data preparation and preprocessing

We split the data into training, validation and evaluation sets. $70\%$ of the whole dataset is used in training set, while the rest is for validation ($15\%$) and evaluation ($15\%$). To ensure the dataset is balanced we shuffle the dataset before the training.

### Data augmentation

Write your answer here

## Model Considerations

We need a model capable of learning a linear relationship.

### Suitable Loss Functions

**MSE and MAE Loss:** The standard loss for regression. 


### Selected Loss Function

**MSE Loss**.
It is the standard default for regression problems and generally leads to faster convergence for data with Gaussian noise.

### Possible architectures

* **Single Linear Layer:** A single neuron with a linear activation (or no activation). This is mathematically equivalent to Classical Linear Regression.
* **Multi-Layer Perceptron (MLP):** A network with hidden layers and non-linear activations.

*Selected Architecture:* **Multi-Layer Perceptron**. Since the target function is not linear, a single layer isn't sufficient.

### Last layer activation

As we want values from -inf to inf we ashall use an **identity layer** as final activation.

### Other Considerations

Write your answer here

## Training

Write your answer here

### Training hyperparameters

Write your answer here

### Loss function graph

![image](../../outs/exercise_02/loss_plot.png)

### Discussion of the training process

Write your answer here

## Evaluation

### Evaluation metrics

Write your answer here

![image](../../outs/exercise_02/train_regression_plot.png)

![image](../../outs/exercise_02/validation_regression_plot.png)

![image](../../outs/exercise_02/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_02/metrics.png)

### Evaluation results

Here you have examples of evaluation results for train, validation and test sets.

Example for train set:

![image](../../outs/exercise_02/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_02/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_02/test_data_points_plot.png)


### Discussion of the results

How the model solves the problem?
Is there overfitting, underfitting or any other issues? 
How can we improve the model?
How this model will generalize to new data?

## Design Feedback loops

Describe the process you have followed to improve the model and the evolution of performance of the model during the process.

You can include a table stating the chanched parameters and the obtained results after the process.


## Questions

Pleaser answer the following questions. Include graphs if necessary. Store the graphs in the `outs/exercise_02` folder.

### Which are the differences you found between previous model and this one?

### Does the model generalizes well to new data?






