# c4ai

A reinforcement learning artificial intelligence system used to play the game
of connect-4.

## How good it is

Good enough to smash all my friends, myself included pretty much all the time.

## How bad it is

Bad enough to lose to a perfect player sometimes.

## Known issues

- `schwi.py` might not work on every system. This problem is a huge pain and
I'm not interested in fixing it at the moment.
- Specifying training arguments in `do_selfplay` of `selfplay_v2.py` doesn't
change anything. I'm too lazy to do do that right now. Will definitely work on
it later though...

## By the way

- `training_pipeline.py` will not use `selfplay_v2.py` by default. However,
I have managed to (and so can you) build the binary on Linux. Use
onnxruntime version 1.2.0 (see [here](https://github.com/microsoft/onnxruntime))
and recent version of GSL.
- If you have something to add, cough it up.
