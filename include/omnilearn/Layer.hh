// Layer.hh

#ifndef OMNILEARN_LAYER_HH_
#define OMNILEARN_LAYER_HH_

#include "json.hh"
#include "Neuron.hh"
#include "ThreadPool.hh"

using json = nlohmann::json;



namespace omnilearn
{



struct LayerParam
{
    LayerParam():
    size(8),
    maxNorm(32),
    distrib(Distrib::Normal),
    mean_boundary(distrib == Distrib::Normal ? 0 : 6),
    deviation(2),
    useOutput(true),
    k(1),
    aggregation(Aggregation::Dot),
    activation(Activation::Relu)
    {
    }

    size_t size; //number of neurons
    double maxNorm;
    Distrib distrib;
    double mean_boundary; //mean (if normal), boundary (if uniform)
    double deviation; //deviation (if normal) or useless (if uniform)
    bool useOutput; // calculate boundary/deviation by taking output number into account
    size_t k; //number of weight set for each neuron (for maxout)
    Aggregation aggregation;
    Activation activation;
};



class Layer
{
    friend void to_json(json& jObj, Layer const& layer);
    friend void from_json(json const& jObj, Layer& layer);

public:
    Layer() = default; // only used in NetworkIO::loadCoefs(). DO NOT USE MANUALLY
    Layer(LayerParam const& param);
    void init(size_t nbInputs, size_t nbOutputs, std::mt19937& generator);
    void init(size_t nbInputs);
    Matrix process(Matrix const& inputs, ThreadPool& t) const;
    Vector processToLearn(Vector const& input, double dropout, double dropconnect, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen, ThreadPool& t);
    Vector processToGenerate(Vector const& input, ThreadPool& t);
    void computeGradients(Vector const& inputGradient, ThreadPool& t);
    void computeGradientsAccordingToInputs(Vector const& inputGradient, ThreadPool& t);
    void keep();
    void release();
    Vector getGradients(ThreadPool& t); //one gradient per input neuron
    void updateWeights(double learningRate, double L1, double L2, Optimizer opti, double momentum, double window, double optimizerBias, ThreadPool& t);
    void updateInput(Vector& input, double learningRate);
    size_t size() const;
    std::vector<std::pair<Matrix, Vector>> getWeights(ThreadPool& t) const;
    void resize(size_t neurons);
    size_t nbWeights() const;

private:
    LayerParam _param;
    size_t _inputSize;
    std::vector<Neuron> _neurons;
};



void to_json(json& jObj, Layer const& layer);
void from_json(json const& jObj, Layer& layer);



} //namespace omnilearn



#endif //OMNILEARN_LAYER_HH_