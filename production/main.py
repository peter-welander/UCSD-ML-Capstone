import argparse
import tensorflow as tf
import os.path
import pathlib

from model import Model, Fit, OpenDataset
from score import f1_score

USAGE = '''
This controls the training and deployment of the flower classifier model.

The follow actions are supported:
 * train - trains the model and stores the weights inside a checkpoint.
   --epochs(e): specify number of epochs to run.
   --continue(c): load the previous weights from a checkpoint before training.
 * eval - load the model from the checkpoint and evaluate the F1 score.
 * save - saves the model.
 * --dest: the destination for the saved model.

'''

IMAGES_DIR = "../images/Flowers5/"


def train(epochs=10, load_checkpoint=False, images_dir=IMAGES_DIR):
    train_ds, val_ds = OpenDataset(images_dir)
    model = Model(len(train_ds.class_names), load_checkpoint=load_checkpoint)
    Fit(model, train_ds, val_ds, epochs=epochs)

def eval(images_dir=IMAGES_DIR):
    train_ds, val_ds = OpenDataset(images_dir)
    model = Model(len(train_ds.class_names), load_checkpoint=True)
    score = f1_score(model, val_ds)
    print("F1 score was: ", score)

def save_model(dest, images_dir=IMAGES_DIR):
    train_ds, val_ds = OpenDataset(images_dir)
    model = Model(len(train_ds.class_names), load_checkpoint=True)
    model.save(dest)

def main(args):
    match args.action:
        case "train":
            train(epochs=args.e, load_checkpoint=args.c, images_dir=args.src)
        case "eval":
            eval(images_dir=args.src)
        case "save":
            dest = args.d
            save_model(dest, images_dir=args.src)
        case _:
            print("'"+ args.action + "' is not a supported action.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='cli',
        description = USAGE,
    )
    parser.add_argument("action")
    parser.add_argument("-c", "-continue",
                        action=argparse.BooleanOptionalAction,
                        help="load weights from the checkpoint before training",
    )
    parser.add_argument("-e", "-epochs", default=10, type=int,
                        help='number of epochs to run',
    )
    parser.add_argument("-d", "-dest", help="the destination to save the stored model.")
    parser.add_argument("-src", default="../images/Flowers/",
                        help="the relative destination of the training data")
    
    args = parser.parse_args()

    main(args)
