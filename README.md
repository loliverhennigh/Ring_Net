# Ring_Net
A Neural Network for compressing dynamical systems into Markov Chains and the like 

# Basics
I want map high dimensional dynamical systems onto low dimensional neural networks. The basic idea is to have a encoder f, compressed system t' and decoder g that will map a system to a low dimensional space and have t' mimic the dynamics. For reasons that are hard to explain I need to train each piece together and so there is fairly complicated learning schedule implemented. Currently the dynamical system is a video of a bouncing ball (28x28x4). I can map this to a 64 vector with minimal error. Still very much a working progress.

# TODOs
- get Markov training well
- implement lstm as t'
- get video systems working
- upload goldfish dataset

