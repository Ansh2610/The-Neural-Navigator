# Neural Navigator

A neural network that learns to navigate through a 2D map. Give it an image with shapes and tell it where to go ("Go to the Red Circle"), and it figures out a path to get there.

## The Problem

Input: A 128x128 image with colored shapes + a text command  
Output: 10 waypoints forming a path from center to the target

## How I Approached It

The model needs to understand both the image (where things are) and the text (what to look for). So I built two separate encoders:

- **Vision encoder**: A CNN that looks at the image and learns to recognize shape positions
- **Text encoder**: Converts words into numbers the model can work with

These two get combined and fed into a decoder that outputs the path coordinates.

## Why These Choices

**CNN for images** - Standard approach for learning spatial features. 4 layers was enough given the simple images.

**Simple word embeddings** - Since there's only 9 possible commands ("Go to the [color] [shape]"), I didn't need anything fancy. Just mapped each word to a learnable vector.

**Normalized coordinates** - Instead of predicting pixel values (0-128), I normalized everything to 0-1. Makes training more stable.

**MSE loss** - Straightforward choice for coordinate regression. Penalizes predictions that are far from ground truth.

## Running It

```
python train.py      # trains the model
python predict.py    # generates path predictions on test images
```

## Challenges

The model learns the right direction but tends to predict shorter paths than the ground truth. This is a known limitation of MSE loss - predicting "average" values minimizes error even if the path doesn't fully reach the target.

## Results

Training converges well. The predicted paths point toward the correct targets, though they're more conservative than the training data paths.
