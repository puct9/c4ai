# c4ai

A reinforcement learning artificial intelligence system used to play the game
of connect-4.

## How good it is

Good enough to smash all my friends, myself included pretty much all the time.

## How bad it is

Bad enough to get smashed by a perfect player every time.

## Known issues

- `schwi.py` might not work on every system. This problem is a huge pain and
I'm not interested in fixing it at the moment.
- Specifying training arguments in `do_selfplay` of `selfplay_v2.py` doesn't
change anything. I'm too lazy to do do that right now. Will definitely work on
it later though...

## By the way

- `training_pipeline.py` will not use `selfplay_v2.py` by default. At least
not until I make finishing touches to the `C4UCT` binary. I'm no big C++ dev
so doing this is half messing around and half torture. You can make it use
`selfplay_v2.py` by changing line 15 `from selfplay import do_selfplay` to
import from `selfplay_v2`. Be aware of the issues I mentioned above though.
- If you have something to add, cough it up.
