// Neuron.hh

#ifndef OMNILEARN_NEURON_HH_
#define OMNILEARN_NEURON_HH_

#include <random>

#include "Activation.hh"
#include "Aggregation.hh"
#include "json.hh"
#include "optimizer.h"

using json = nlohmann::json;



namespace omnilearn
{



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
    void updateWeights(double learningRate, double L1, double L2, double weightDecay, double maxNorm, bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, bool lockWeights, bool lockBias, bool lockParametric);
    //one gradient per input neuron
    Vector getGradients() const;
    void keep();
    void release();
    void computeGradientsAccordingToInputs(double inputGradient);
    void updateInput(Vector& input, double learningRate); // for generation
    void resetGradientsForGeneration();
    size_t inputSize() const;
    std::pair<double, double> L1L2() const;
    size_t getNbParameters() const;

private:
    std::unique_ptr<IAggregation> _aggregation;
    std::unique_ptr<IActivation> _activation;

    Matrix _weights;
    Vector _bias;

    Vector _input;
    std::pair<double, size_t> _aggregResult;
    double _actResult;

    double _actGradient; //gradient of activation according to aggregation result
    Matrix _gradients; //sum (over features of the batch) of partial gradient of aggregation according to each weight
    Vector _biasGradients;
    Vector _featureGradient; // store gradients for the current feature (for the previous layer's neurons)
    std::vector<size_t> _weightsetCount; //counts the number of features passed in each weight set
    Matrix _previousWeightGradient; // used for optimizers (momentum effect)
    Vector _previousBiasGradient; // used for optimizers (momentum effect)
    Matrix _previousWeightGradient2; // used for optimizers (window effect)
    Vector _previousBiasGradient2; // used for optimizers (window effect)
    Matrix _optimalPreviousWeightGradient2; // used for optimizers (see AMSGrad documentation)
    Vector _optimalPreviousBiasGradient2; // used for optimizers (see AMSGrad docimentation)
    Matrix _previousWeightUpdate; // used for optimizers (to replace learning rate)
    Vector _previousBiasUpdate; // used for optimizers (to replace learning rate)

    Matrix _savedWeights;
    Vector _savedBias;

    Vector _generativeGradients; // partial gradient for each input to tweak
};



void to_json(json& jObj, Neuron const& neuron);
void from_json(json const& jObj, Neuron& neuron);



} //namespace omnilearn



#endif //OMNILEARN_NEURON_HH_