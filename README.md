# UCSD ML Capstone Project

## Summary

For my capstone I built a flower classifier that can identify ~300 types of flowers. 

## Dataset

I used the following dataset: https://www.kaggle.com/datasets/bogdancretu/flower299

## Code layout

* The code for training the model is in the [production](production) directory. \
See the README for how to train the model.
* The app is in the [app](app) directory.

## Research steps

1. [Project Ideas](https://docs.google.com/document/d/1OwoVThBr-NEvzDFm6YNk9EviZhY2S7FuJz6icmsn5NU/edit?usp=sharing)
1. [Testing AWS](https://docs.google.com/document/d/1C1HcN_NeIyYbGJT9znWVsDQfOVE8tayJ222FErkRPOQ/edit#heading=h.ntgqoiq6rz4a)
1. [Project Proposal](Project%20Proposal.pdf)
1. [Data collection](Step1-Reading-Data.ipynb)
1. [Exploratory Data Analysis](Step2-EDA.ipynb)
1. [Reproduce research](Step3-Reproduce-Research.ipynb)
1. [Experiment with various models](Step7-Picking-Base-Model.ipynb)
1. [Prototyping](Step8-Prototype.ipynb)
1. [Scaling up](Step9-Scale-Up.ipynb)
1. [Study advanced units](Step10-Study-Advanced-Units.ipynb)
1. [Pick deployment method](https://docs.google.com/document/d/1wbKFaBcFDp6WvjLrykD911FqIPU1Ze3CqbnJgVaP0_E/edit#heading=h.sdgrs03vb91x)
1. [Design deployment architecture](https://docs.google.com/document/d/1sr8bqBBqgjXQrLTXV3ZCatJyUzaGcyN55LnAQDyXClE/edit#heading=h.encavr6phqug)

## Learnings


* Exploratory data analysis is important. When scaling up and looking at my confusion matrix I realized that some flowers were labelled both with their scientific name and their common name. This made it impossible for the model to classify them correctly. Well, the model might label a flower correctly with its scientific name, but the validation data might have its common name... Had to go and fix my data to improve the model.

* Accurately classifying many flower types is a hard problem. Scaling from 15 to 30 flower types more than doubles the training time and required making the model more complicated. I ended up using checkpointing to work around this problem.

* Transfer learning is very helpful. In particular I found the [BiT](https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html) models to be very useful in speeding up the training of my model.

## Final Product

My app is deployed using StreamLit. To access it please visit this [url](https://ucsd-ml-capstone-97iasbglz7.streamlit.app/)