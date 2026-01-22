# Neural Navigator

Neural network that takes a 2D map image and a text command ("Go to the Red Circle") and outputs a path to the target.

## How to Run

```
python train.py      # train the model
python predict.py    # run predictions on test images
```

## Challenges & Solutions

### What was the hardest part?

Getting the paths to actually reach the targets. The model kept predicting coordinates close to the center of the image instead of extending toward the shapes. This happens because MSE loss doesn't care if you're conservative - predicting middle-ish values gives decent error scores even if the path is too short.

I tried different things: added BatchNorm, tried Dropout, played with learning rates. Eventually switched to a Transformer decoder which helped a bit with learning the sequence of waypoints, but the core issue is really the loss function not being ideal for path prediction.

### A bug I ran into

Early on the model was ignoring the text command completely - paths would go random directions regardless of what shape I told it to find.

Turned out my tokenizer was case-sensitive. The vocabulary had "Red" but the input text was "red" (lowercase). So everything was getting mapped to the padding token.

Debugging steps:
1. Noticed loss was still decreasing, so model was learning something
2. Visualized predictions - saw they didn't correlate with text at all
3. Added print statements in data loader to see actual token values
4. Found all tokens were 0 (padding)
5. Fixed by adding `.lower()` when looking up words

Took me like 20 min to find a one-line fix.

### Results

- Best validation MSE: ~0.016
- Model correctly identifies which shape to go to based on text
- Paths point in right direction but are shorter than ground truth
