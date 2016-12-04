Robust PG
========================

To run, do the following:

    pip install -r requirements.txt
    python -m robustpg

This will run 100 trials of mirrored, sparse, and normal REINFORCE on the noisy Cart Pole environment and output the performance plot. This can be changed in the main `robustpg/__main__.py` file.
