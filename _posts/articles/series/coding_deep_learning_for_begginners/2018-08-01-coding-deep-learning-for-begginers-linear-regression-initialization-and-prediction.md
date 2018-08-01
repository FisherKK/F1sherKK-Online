---
layout: post
author: Kamil Krzyk
title:  "Coding Deep Learning for Beginners — Linear Regression (Part 1): Initialization and Prediction"
date:   2018-08-01 08:00:00 +0200
mathjax: true
comments: true
categories: machine_learning linear_regression pandas
permalink: article/coding_deep_learning_series/:year/:month/:day/:title
---
This is the 3nd article of series **“Coding Deep Learning for Beginners”**. You will be able to find here links to all articles, agenda, and general information about an estimated release date of next articles **on the bottom**. They are also available in [my open source portfolio — MyRoadToAI](https://github.com/FisherKK/F1sherKK-MyRoadToAI), along with some mini-projects, presentations, tutorials and links.

You can also read the article on Medium.

•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•
{: .separator}

## Why Linear Regression?
Some of you may wonder, why the article series about explaining and coding Neural Networks starts with **basic Machine Learning algorithm** such as Linear Regression. It's very justifiable to start from there. First of all, it is a very plain algorithm so the reader can grasp an **understanding of fundamental Machine Learning concepts** such as *Supervised Learning*, *Cost Function*, and *Gradient Descent*. Additionally, after learning Linear Regression it is quite easy to understand Logistic Regression algorithm and believe or not - it is possible to categorise that one as small Neural Network. You can expect all of those and even more covered in few next articles!

{: .header_padding_top}
## Tools
Let's introduce the **most popular libraries** that can be found in every Python based Machine Learning or Data Science related project.
- [NumPy](http://www.numpy.org/) - a library for scientific computing, perfect for Multivariable Calculus & Linear Algebra. Provides [ndarray](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.html) class which can be compared to **Python list that can be treated as vector or matrix**.
- [Matplotlib](https://matplotlib.org/) - toolkit for **data visualisation**, allows to create various 2d and 3d graphs.
- [Pandas](https://pandas.pydata.org/) - this library is a wrapper for Matplotlib and NumPy libraries. It provides [DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) class. It **treats NumPy matrices as tables**, allowing access to rows and columns by their attached names. Very helpful in **data loading, saving, wrangling, and exploration process**. Provides an interface of functions that makes deployment faster.

Each library can be installed separately with using [Python PyPi](https://pypi.org/project/pip/). They will be imported in code of every article under following aliases.

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
{% endhighlight %}

{: .header_padding_top}
## What is Linear Regression?
It’s a **Supervised Learning algorithm** which goal is to **predict continuous, numerical values based on given data input**. From the geometrical perspective, each data sample is a point. Linear Regression tries to **find parameters of the linear function**, so the **distance between the all the points and the line is as small as possible**. Algorithm used for parameters update is called **Gradient Descent**.

<img src="https://www.dropbox.com/s/dowd08tzrpxz0co/a3_linear_train_gif.gif?dl=1" width="600px">{: .image-center .image-offset-top}

{: .image-caption }
*Training of Linear Regression model. The left graph displays the change of linear function parameters over time. The plot on the right renders the linear function using current parameters (source: [Siraj Raval GitHub](https://github.com/llSourcell/linear_regression_live)).*

For example, if we have a dataset consisting of apartments properties and their prices in some specific area, Linear Regression algorithm can be used to find a mathematical function which will try to estimate the value of different apartment (outside of the dataset), based on its attributes.

<img src="https://www.dropbox.com/s/7xz0r73i3a2yvbc/a3_linear_example_1.png?dl=1" width="600px">{: .image-center .image-offset-top}

Another example can be a prediction of food supply size for the grocery store, based on sales data. That way the business can decrease unnecessary food waste. **Such mapping is achievable for any correlated input-output data pairs.**

<img src="https://www.dropbox.com/s/6r7k49tyo2a77ls/a3_linear_example_2.png?dl=1" width="600px">{: .image-center .image-offset-top}

{: .header_padding_top}
### Data preparation
Before coding Linear Regression part, it would be good to have some problem to solve. It is possible to find a lot of datasets on websites like [UCI Repository](https://archive.ics.uci.edu/ml/index.php) or [Kaggle](https://www.kaggle.com/). After going through many of those, none was suitable for study case of this article.

In order to get data, I’ve entered Polish website [dominium.pl](https://www.dominium.pl/pl/szukaj/mieszkania/nowe), which is a search engine for apartments in Cracow city — area where I live. I have entirely randomly chosen 76 apartments, written down their attributes and saved to the **.csv file**. The goal was to **train Linear Regression model capable of predicting apartments prices** in Cracow.

Dataset is available on my Dropbox under this [link](https://www.dropbox.com/s/1octs0jg5o5j82o/cracow_apartments.csv?dl=0).

{: .header_padding_top}
### Loading data
Let’s start by reading data from the .csv file to DataFrame object of Pandas and displaying a few data rows. To achieve that [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) function will be used. Data is separated with colon character which is why `sep=","` parameter was added. Function [head](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html) renders first five rows of data in the form of the pleasantly readable HTML table.

{% highlight python %}
df_data = pd.read_csv("cracow_apartments.csv", sep=",")
df_data.head()
{% endhighlight %}

The output of the code looks as following:

<img src="https://www.dropbox.com/s/vbymhzew5izwjt8/a3_dataframe_head_appartments.png?dl=1" width="350px">{: .image-center .image-offset-top}

{: .image-caption }
*DataFrame visualisation in Jupyter Notebook.*

As presented in the table, there are **four features** describing apartment properties:

- **distance_to_city_center** - distance from dwelling to [Cracow Main Square](https://en.wikipedia.org/wiki/Main_Square,_Krak%C3%B3w) on foot, measured with Google Maps,
- **rooms** - the number of rooms in the apartment,
- **size** - the area of the apartment measured in square meters,
price - target value (the one that needs to be predicted by model), cost of the apartment measured in Polish national currency — [złoty](https://en.wikipedia.org/wiki/Main_Square,_Krak%C3%B3w).

{: .header_padding_top}
### Visualising data
It is very important to always understand the structure of data. The more features there are, the harder it is. In this case, [scatter plot](https://en.wikipedia.org/wiki/Scatter_plot) is used to **display the relationship between target and training features**.

<img src="https://www.dropbox.com/s/dvkxyx55o3e14ga/a3_features_presented.png?dl=1" width="1000px">{: .image-center .image-offset-top}

{: .image-caption }
*Charts show whole data from cracow_apartments.csv. It was prepared with Matplotlib library in Jupyter Notebook. The code used to create these charts can be found under this [link](https://medium.com/r/?url=https%3A%2F%2Fgist.github.com%2FFisherKK%2F0113b1eda361856a1cd29ad4fbd180d2).*

Depending on what is necessary to show, some other types of visualization (e.g. [box plot](https://en.wikipedia.org/wiki/Box_plot)) and techniques could be useful (e.g. [clustering](https://en.wikipedia.org/wiki/Cluster_analysis)). Here, a **linear dependency between features can be observed** — with the increase of values on axis x, values on the y-axis are linearly increasing or decreasing accordingly. It’s great because if that was not the case (e.g. relationship would be exponential), then it would be hard to fit a line through all the points and different algorithm should be considered.

{: .header_padding_top}
## Formula
The Linear Regression **model is a mathematical formula** that takes **vector of numerical values** (attributes of single data sample) as an input and uses them to **make a prediction**.

Mapping the same statement in the context of the presented problem, there are 76 samples containing attributes of Cracow apartments where each sample is a vector from mathematical perspective. Each **vector of features is paired with target value** (expected result from formula).

<img src="https://www.dropbox.com/s/m8ysc6mqpditnrf/a3_features.png?dl=1" width="800px">{: .image-center .image-offset-top}

According to the algorithm, **every feature has a weight parameter assigned**. It represents it’s **importance** to the model. The goal is to find the values of weights so the following equation is met for every apartment data.

<img src="https://www.dropbox.com/s/5s9pi5cfwb99u8c/a3_features_in_formula_nob.png?dl=1" width="800px">{: .image-center .image-offset-top}

The left side of the equation is a **linear function**. As **manipulation of weight values can change an angle of the line**. Although, there is a still one element missing. Current function is always going through (0,0) point of the coordinate system. To fix that, another trainable parameter is added.

<img src="https://www.dropbox.com/s/xm5oz60mstbkna7/a3_features_in_formula_withb.png?dl=1" width="800px">{: .image-center .image-offset-top}

The parameter is named **bias and it gives the formula a freedom to move on the y-axis up and down**.
The purple parameters belong to the model and are used for prediction for every incoming sample. That's why finding a solution that works best for all samples is necessary. Formally the formula can be written as:

$$f(x) = w_0 x_0 + w_1 x_1 + ... + w_n x_n + b$$

where:
- $x$ - vector of data used for prediction or training
- $n$ - number of features in vector
- $x_i$ - i-th feature of vector $x$
- $w_i$ - i-th weight attached to i-th feature of vector $x$
- $b$ - bias

{: .header_padding_top}
## Initialization
It's a phase where the **first version of a model is created**. Model after initialization can already be used for prediction but without training process, the results will be far from good. There are two things to be done:
- **create variables in code** that represents weights and bias parameters,
- **decide on starting values** of model parameters.

Initial values of model parameters are very crucial for Neural Networks. In case of Linear Regression **parameter values can be set to zero** at the start.

{% highlight python %}
def init(n):
    return {"w": np.zeros(n), "b": 0.0}
{% endhighlight %}

Function `init(n)` returns a dictionary containing model parameters. According to the terminology presented in the legend below the mathematical formula, *n is the number of features* used to describe data data sample. It is used by [zeros](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.zeros.html) function of NumPy library, to return a vector of ndarray type with n elements and zero value assigned to each. Bias is a scalar set to 0.0 and **it is a good practice to keep the variables as floats rather than integers**. Both weights and bias are accessible under “w” and “b” dictionary keys accordingly.

For Cracow apartment dataset there are three features describing each sample. Here is the result of calling `init(3)`:

<img src="https://www.dropbox.com/s/vxqalm67gfdnf2r/a3_parameters.png?dl=1" width="350px">{: .image-center .image-offset-top}

{: .header_padding_top}
## Prediction
Created model parameters can be used by the model for making a prediction. The formula has been already shown. Now it’s time to turn it into the Python code. First, every feature has to be multiplied by its corresponding weight and summed up. Then bias parameter needs to be added to the product of the previous operation. The outcome is a prediction.

{% highlight python %}
def predict(x, parameters):
    # Prediction initial value
    prediction = 0

    # Adding multiplication of each feature with it's weight
    for weight, feature in zip(parameters["w"], x):
        prediction += weight * feature

    # Adding bias
    prediction += parameters["b"]

    return prediction
{% endhighlight %}

Function `predict(x, parameters)` takes two arguments:
- vector `x` of features representing a data sample (e.g. single apartment),
- Python dictionary `parameters` which stores parameters of the model along with their current state.

{: .header_padding_top}
## Assemble
Let’s put together all code parts that were created and display at the results.

{% highlight python %}
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data from .csv
df_data = pd.read_csv("cracow_apartments.csv", sep=",")

# Used features and target value
features = ["size"]
target = ["price"]

# Slice Dataframe to separate feature vectors and target value
X, y = df_data[features].as_matrix(), df_data[target].as_matrix()

# Initialize model parameters
n = len(features)
model_parameters = init(n)

# Make prediction for every data sample
predictions = [predict(x, model_parameters) for x in X]
{% endhighlight %}

**Only one feature was used for prediction** what reduced formula to form:

<img src="https://www.dropbox.com/s/dmr34wh6d0b9mbs/a3_formula_reduced.png?dl=1" width="800px">{: .image-center .image-offset-top}

This was intentional as **displaying results on the data which has more than 1–2–3 dimensions becomes troublesome**, unless [Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) techniques are used (e.g. [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)). From now on, for learning purposes all code development will be done only on **size** feature. When Linear Regression code will be finished, results with usage of all features will be presented.

<img src="https://www.dropbox.com/s/m0bvstis6nure6u/a3_model_projection_all_zeros.png?dl=1" width="600px">{: .image-center .image-offset-top}

{: .image-caption }
*Line used to fit the data by Linear Regression model with current parameters. Code for visualisation is available under this [link](https://gist.github.com/FisherKK/a78a54d4fa9bdd56c9512f24b98df5f9).*

The model parameters were initialized with zero values which means that the output of the formula will always be equal to zero. Consequently, the `prediction` is a Python list of 76 zero values which are predicted prices for each apartment separately. But that’s ok for now. **Model behavior will improve after training with the Gradient Descent is used and explained.**

Bonus takeouts from the code snippet are:
- Features to be used by model and target value were stored in `features` and `target` Python lists. Thanks to that there is no need to modify the whole code if a different set of features should be used.
- It is possible to parse DataFrame object to ndarray by using [as_matrix](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html) function.

•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•
{: .separator}

{: .header_padding_top}
## Summary
In this article I have introduced the tools that I am going to use in the whole article series. Then I have presented the problem I am going to solve with Linear Regression algorithm. At the end, I have shown how to create Linear Regression model and use it for making a prediction.

In the next article I will explain how to compare sets of parameters and measure model performance. Finally, I will show how to update model parameters with Gradient Descent algorithm.

## Next Article
You can expect the next article on the **08.08.2018**.
