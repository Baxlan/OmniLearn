// Neuron.hh

#ifndef OMNILEARN_NEURON_HH_
#define OMNILEARN_NEURON_HH_

#include <memory>
#include <random>

#include "Activation.hh"
#include "Aggregation.hh"
#include "json.hh"

using json = nlohmann::json;



namespace omnilearn
{



enum class Optimizer {None, Momentum, Nesterov, Adagrad, Rmsprop, Adam, Adamax, Nadam, AmsGrad};
enum class Distrib {Uniform, Normal};



class Neuron
{
    friend void to_json(json& jObj, Neuron const& neuron);
    friend void from_json(json const& jObj, Neuron& neuron);

public:
    Neuron() = default; // only used in from_json(Layer). Segfault are possible if used manually
    Neuron(Aggregation aggregation, Activation activation);
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
    void keep();
    void release();
    void computeGradientsAccordingToInputs(double inputGradient);
    void updateInput(Vector& input, double learningRate);
    //first is weights, second is bias
    std::pair<Matrix, Vector> getWeights() const;
    size_t nbWeights() const;
    void setAggrAct(Aggregation aggr, Activation act);

private:
    std::shared_ptr<IAggregation> _aggregation;
    std::shared_ptr<IActivation> _activation;

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



void to_json(json& jObj, Neuron const& neuron);
void from_json(json const& jObj, Neuron& neuron);



} //namespace omnilearn



#endif //OMNILEARN_NEURON_HH_