# Logistic Regression & Log-Loss Workshop

**Group:** Group 3  
**Use case:** Predicting Pass/Fail from Study Hours

## Project Overview

In this workshop, we explored how **Logistic Regression** can predict whether a student passes or fails an exam based on their study hours. We also implemented **Binary Log-Loss (Cross-Entropy)** ourselves to understand how classification models measure prediction errors.

The project includes a pair-programming and peer-review report. We compared our approach with another team's implementation to better understand how model training and Log-Loss work.

## Learning Objectives

- Understand binary classification using `Pass = 1` and `Fail = 0`.
- Train a Logistic Regression model with Scikit-Learn.
- Calculate probabilities of passing and convert them into classifications using a 0.5 threshold.
- Implement Binary Log-Loss using NumPy.
- Validate our Log-Loss calculation against Scikit-Learn.
- Analyze individual losses and visualize predictions and errors.
- Discuss the strengths and limitations of our approach during peer review.


## Requirements

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

## How to Run

1. Clone our repository:

   ```bash
   git clone https://github.com/jcamilochirivi/LogisticRegressionClassifier_Workshop.git
   ```

2. Open the project folder:

   ```bash
   cd LogisticRegressionClassifier_Workshop
   ```

3. Install the dependencies using the command above.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `PairProgrammingReport.ipynb` and select **Run All** to execute the code cells in order.

The study-hours dataset is defined inside the notebook, so there is **no external CSV file to download**.

## Methodology

### 1. Dataset

We used the 19 study-hours and Pass/Fail observations provided in the original workshop. The input feature is the number of hours studied, and the target label indicates whether a student passed (`1`) or failed (`0`).

### 2. Model Training

We used Scikit-Learn's `LogisticRegression` and `model.fit(X, y)` to train our classifier. The trained model estimates each student's probability of passing. We used a threshold of 0.5 to obtain the final Pass/Fail prediction.

### 3. Manual Log-Loss Implementation

We implemented `binary_log_loss()` to compare actual results with predicted probabilities. The function uses `np.clip` to prevent numerical issues from trying to compute the logarithm of zero.

We validated our manual calculation with `sklearn.metrics.log_loss` and obtained the same result at the displayed precision.

### 4. Results and Visualizations

The notebook includes a table of predictions, individual losses, a probability curve, a chart of individual errors, and simple examples demonstrating how confident incorrect predictions receive higher penalties.

## Results

| Metric | Result |
|---|---:|
| Training observations | 19 |
| Training accuracy | 89.5% (17 of 19) |
| Manual average Log-Loss | 0.275395 |
| Scikit-Learn average Log-Loss | 0.275395 |
| Difference at displayed precision | 0.0000000000 |

**How to interpret the results:** Accuracy measures how many Pass/Fail classifications were correct, while Log-Loss measures the quality of the predicted probabilities. A lower Log-Loss generally indicates better probability predictions on the same evaluation data. It is **not** an error percentage.

**Limitation:** These numbers were calculated using the same observations used to train the model. They describe training performance, not guaranteed performance on new students. A separate test set would be needed to evaluate generalization.

## Pair Programming and Peer Review

Our notebook also documents our discussion points and review of another team's solution. Our team used Scikit-Learn for model training and implemented Log-Loss manually. The peer team's approach included manual gradient-descent training, allowing us to compare two ways of learning about Logistic Regression.

The peer-review sections cover:

- Our three talking points and possible questions about our implementation.
- Technical feedback on our work.
- Observations about the other team's code and approach.
- Comparison of model training methods and results.
- Learning reflections and limitations.

The peer team's code remains external to this repository. Any discussion notes requiring confirmation from the live peer-review session should be finalized using the team's actual feedback.

## Key Takeaways

We learned that Logistic Regression predicts probabilities, whereas Log-Loss evaluates how good those probabilities are. We also learned that Log-Loss penalizes confident wrong predictions more heavily, and that `np.clip` makes the calculation numerically stable. Comparing our Scikit-Learn approach with a manual gradient-descent implementation helped us better understand model training.

## Reference

Original workshop and supporting materials:  
[ProfEspinosaAIML – LogisticRegressionClassifier_Workshop](https://github.com/ProfEspinosaAIML/LogisticRegressionClassifier_Workshop.git)
