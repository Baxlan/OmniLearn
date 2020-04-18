// Neuron.hh

#ifndef OMNILEARN_NEURON_HH_
#define OMNILEARN_NEURON_HH_

#include <memory>
#include <random>

#include "Activation.hh"
#include "Aggregation.hh"



namespace omnilearn
{



enum class Optimizer {None, Momentum, Nesterov, Adagrad, Rmsprop, Adam, Adamax, Nadam, AmsGrad};
enum class Distrib {Uniform, Normal};



class Neuron
{
public:
    Neuron(size_t aggregation, size_t activation);
    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, size_t k, std::mt19937& generator, bool useOutput);
    //each line of the input matrix is a feature. Returns one result per feature.
    Vector process(Matrix const& inputs) const;
    double processToLearn(Vector const& input, double dropconnect, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen);
    double processToGenerate(Vector const& input);
    //compute gradients for one feature, finally summed for the whole batch
    void computeGradients(double inputGradient);
    void updateWeights(double learningRate, double L1, double L2, double maxNorm, Optimizer opti, double momentum, double window, double optimizerBias);
    //one gradient per input neuron
    Vector getGradients() const;
    void save();
    void loadSaved();
    void computeGradientsAccordingToInputs(double inputGradient);
    void updateInput(Vector& input, double learningRate);
    //first is weights, second is bias
    std::pair<Matrix, Vector> getWeights() const;
    rowVector getCoefs() const;
    size_t nbWeights() const;
    void setCoefs(Matrix const& weights, Vector const& bias, Vector const& aggreg, Vector const& activ);


protected:
    std::shared_ptr<AggregationFunc> _aggregation;
    std::shared_ptr<ActivationFct> _activation;

    Matrix _weights;
    Vector _bias;

    Vector _input;
    std::pair<double, size_t> _aggregResult;
    double _actResult;

    double _actGradient; //gradient between aggregation and activation
    Matrix _gradients; //sum (over features of the batch) of partial gradient for each weight
    Vector _biasGradients;
    Vector _featureGradient; // store gradients for the current feature
    std::vector<size_t> _weightsetCount; //counts the number of gradients in each weight set
    Matrix _previousWeightUpdate;
    Vector _previousBiasUpdate;

    Matrix _savedWeights;
    Vector _savedBias;

    Vector _generativeGradients; // partial gradient for each input to tweak
};



} //namespace omnilearn



#endif //OMNILEARN_NEURON_HH_