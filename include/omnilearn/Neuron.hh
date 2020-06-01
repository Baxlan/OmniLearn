// Neuron.hh

#ifndef OMNILEARN_NEURON_HH_
#define OMNILEARN_NEURON_HH_

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
    Neuron() = default; // only used in from_json(Layer). Segfault may occure if used manually. Cannot be private :(
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
    void resetGradientsForGeneration();
    //first is weights, second is bias
    size_t nbWeights() const;
    std::pair<double, double> L1L2() const;

private:
    std::shared_ptr<IAggregation> _aggregation;
    std::shared_ptr<IActivation> _activation;

    Matrix _weights;
    Vector _bias;

    Vector _input;
    std::pair<double, size_t> _aggregResult;
    double _actResult;

    double _inputGradient; //sum (over features of the batch) of gradients passed in input (used for parametric activation)
    double _sumedActGradient; //sum (over features of the batch) of gradient of activation according to aggregation result (used fir parametric aggregation)
    double _actGradient; //gradient of activation according to aggregation result
    Matrix _gradients; //sum (over features of the batch) of partial gradient of aggregation according to each weight
    Vector _biasGradients;
    Vector _featureGradient; // store gradients for the current feature (for the previous layer's neurons)
    std::vector<size_t> _weightsetCount; //counts the number of features passed in each weight set
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