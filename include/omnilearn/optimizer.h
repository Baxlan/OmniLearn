// optimizer.h

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_



namespace omnilearn
{



enum class Optimizer {None, Momentum, Nesterov, Adagrad, Rmsprop, Adam, Adamax, Nadam, AmsGrad};




void optimizedUpdate(Optimizer opti, double momentum, double window, double optimizerBias);



} // namespace omnilearn



#endif // OPTIMIZER_H_