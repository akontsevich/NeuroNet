# NeuroNet
Simple but very fast C++ back propagation neural network library. Could be used
for approximations, models identification, patterns recognition, multi-factor
prognosis, etc

**Contains:**
  - **TNeuroNet** - neural network class with 3 learning functions 
      - Standard Back propagation error learing function 
      - SCG - scaled conjugate gradient learing function (fastest, smallest SSE)
      - Learing function with genetic algorithm (unique results: SSE on test
      patterns are less than on learning pattern)
      
      and 3 activation functions:
      - Standard logistic activation function (best)
      - Hyperbolic tangent activation function
      - Exponential activation function
  - **TGeneticAlgorithm** - real value coding genetic algorithm
  - **TDataSource** - abstract class containing data source for neuro net learning
  - **TLearnPattern** - class contains normalized data patterns for neuro net learning
  - Some other helper classes
