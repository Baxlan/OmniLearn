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



class Neuron
{
    friend void to_json(json& jObj, Neuron const& neuron);
    friend void from_json(json const& jObj, Neuron& neuron);

public:
    Neuron() = default; // only used in from_json(Layer). Segfault may occure if used manually. Cannot be private :(
    Neuron(Aggregation aggregation, Activation activation);
    void init(Distrib distrib, double distVal1, double distVal2, size_t nbInputs, size_t nbOutputs, std::mt19937& generator, bool useOutput);
    //each line of the input matrix is a feature. Returns one result per feature.
    Vector process(Matrix const& inputs) const;
    double processToLearn(Vector const& input, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen);
    double processToGenerate(Vector const& input);
    //compute gradients for one feature, finally summed for the whole batch
    void computeGradients(double inputGradient);
    void updateWeights(double learningRate, double L1, double L2, double weightDecay, double maxNorm, bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, bool lockWeights);
    //one gradient per input neuron
    Vector getGradients() const;
    void keep();
    void release();
    void computeGradientsAccordingToInputs(double inputGradient);
    void updateInput(Vector& input, double learningRate); // for generation
    void resetGradientsForGeneration();
    std::pair<double, double> L1L2() const;
    size_t getNbParameters(bool lockWeights) const;
    Neuron getCopyForOptimalLearningRateDetection() const;

private:
    std::unique_ptr<IAggregation> _aggregation;
    std::unique_ptr<IActivation> _activation;
    Activation _activationType;
    Aggregation _aggregationType;

    Vector _weights;

    Vector _input;
    double _aggregResult;
    double _actResult;
    bool _dropped;
    BoolVector _connectDropped;

    double _actGradient; //gradient of activation according to aggregation result
    Vector _featureGradient; // store gradients for the current feature (for the previous layer's neurons)
    std::vector<LearnableParameterInfos> _weightInfos;
    size_t _count;
    Size_tVector _counts; // one count per weight, useful in case of dropconnect
    Vector _savedWeights;

    Vector _generativeGradients; // partial gradient for each input to tweak
};



void to_json(json& jObj, Neuron const& neuron);
void from_json(json const& jObj, Neuron& neuron);



} //namespace omnilearn



#endif //OMNILEARN_NEURON_HH_