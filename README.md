# NeuroNet
Simple but very fast C++ back propagation neural network library.

**Contains:**
  - **TNeuroNet** - neural network class with 3 learning functions 
      - Standard Back propagation error learing function 
      - SCG - scaled conjugate gradient learing function (very fast)
      - Learing function with genetic algorithm
      
      and 3 activation functions:
      - Standard logistic activation function (best)
      - Hyperbolic tangent activation function
      - Exponential activation function
  - **TGeneticAlgorithm** - real value coding genetic algorithm
  - **TDataSource** - abstract class containing data patterns for neuro net learning
  - Some other helper classes
