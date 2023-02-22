# VRPNet

This is the codebase for my Master's thesis "Prediction of route distances for the Vehicle Routing Problem using Graph Neural Networks".

## Abstract

Many transportation and logistics problems require a quick and accurate estimation of routing distances. The existing approaches for this task are primarily based on linear regression. This thesis proposes a Graph Neural Network model for total route distance estimation in a Vehicle Routing Problem with no constraints. To train the model, multiple datasets of VRP instances are generated and the capacity of the model to generalize to problems of different sizes is explored. Distance labels for the datasets are obtained using a OR-tools. Experimental results indicate a better prediction accuracy when the model is trained on a fixed size instances, whereas better generalization is achieved when training on a dataset of mixed input size.



The Vehicle Routing Problem is a famous problem in combinatorial optimization that aims to find an optimal delivery route for a set of customers. It is an NP-hard problem, and so finding exact solutions is computationally intractable. Classically, many hand-tailored approximate algorithms have been developed to find near-optimal solutions in reasonable time. However, it is challenging to develop new improved algorithms since it requires expert domain knowledge. With recent development in deep learning models, it is enticing to incorporate these in the field of combinatorial optimization to automatically learn improved algorithms. Another attractive aspect of deep learning models are their fast computation times, potentially enabling application to even greater problem sizes than current solvers can handle.

In this thesis, we present a supervised deep learning framework for obtaining approximate solutions to the Vehicle Routing Problem on 2D euclidean graphs. We implement a Graph Neural Network that learns a probabilistic representation of the solution space to the problem. This representation is converted into a valid solution using a beam search decoder. The beam search procedure is parallelized, allowing for fast search over the solution space. The performance of our model is evaluated on three different problem sizes. The network is trained on sets of near-optimal solutions obtained using the OR-tools solver.

The model manages to produce decent results on small problem sizes with 20 nodes, finding solutions under 5 \% from the target on average. Applying the model to larger problem sizes proves challenging, struggling to find good solutions for problems with 50 nodes, being over 100 \% from the target on average. Another weakness of the model is its inability to generalize to problem sizes not seen during training. This restricts our framework to trivially small problem sizes efficiently solvable by standard solvers.

## Necessary libraries

To run the code, you need to make sure you have the following python libraries installed:

* NumPy
* Pytorch
* Scipy
* Scikit-learn
* networkx
* Matplotlib
* Pickle
* ortools
