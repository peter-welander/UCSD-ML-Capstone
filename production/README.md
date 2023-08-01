# README

This folder contains a CLI that can train the flower classifier model.

## Setup

Start by installing the requirements.
```
pip install -r requirements.txt
```

Then install the data by download [this](https://www.kaggle.com/datasets/bogdancretu/flower299) kaggleset.
Extract the data into a parallel directory, i.e. "../images/Flowers", or specify the source of the data with a flag (-src).

## Running the trainer.

To run the trainer:
```
python main.py <args>
```

The follow actions are supported:
 * train - trains the model and stores the weights inside a checkpoint.
   --epochs(e): specify number of epochs to run.
   --continue(c): load the previous weights from a checkpoint before training.
 * eval - load the model from the checkpoint and evaluate the F1 score.
 * save - saves the model.
 * --dest: the destination for the saved mode
