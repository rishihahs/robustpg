Robust PG
========================

To run, do the following:

    pip install -r requirements.txt
    python -m robustpg

This will run 200 trials of mirrored, sparse, and normal REINFORCE on the noisy Cart Pole environment and output the performance plot. This can be changed in the main `robustpg/__main__.py` file.

Files of interest in `robustpg/`:
* `__main__.py` Sets up Cart Pole environment and runs experiments
* `mirrordescent.py` Implmentation of p-norm mirror descent (includes mirror maps, their gradients, Bregman divergence, etc.)
* `pg.py` Main REINFORCE implementation along with modified updates for sparse and mirrored pg
* `policy.py` Implementation of softmax policy
