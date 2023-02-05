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
    size(16),
    maxNorm(0),
    distrib(Distrib::Normal),
    mean(0), //only used if distrib==normal

    //if activation is linear around x=0 (such as sigmoid or tanh):
        //use UNIFORM distrubution and:
        //if using Xavier Weight Initialization:
            //set useOutput to false and deviation_boundary to 1.
        //else, use normalized version of Xavier Weight Initialization:
            //set useOutput to true and deviation_boundary to 6.

    //else if activation is not linear around x=0 (such as relu):
        // use NORMAL distribution, set useOutput to false and set deviation_boundary to 2
    deviation_boundary(2),
    useOutput(false),
    k(1),
    aggregation(Aggregation::Dot),
    activation(Activation::Relu),
    lockWeights(false),
    lockBias(false),
    lockParametric(false)
    {
    }

    size_t size; //number of neurons
    double maxNorm;
    Distrib distrib;
    double mean; //mean (if normal), nor used (if uniform)
    double deviation_boundary; //deviation (if normal) or boundary (if uniform)
    bool useOutput; // calculate boundary/deviation by taking output number into account
    size_t k; //number of weight set for each neuron (for maxout)
    Aggregation aggregation;
    Activation activation;
    bool lockWeights;
    bool lockBias;
    bool lockParametric;
};



class Layer
{
    friend void to_json(json& jObj, Layer const& layer);
    friend void from_json(json const& jObj, Layer& layer);

public:
    static LayerParam generateLinearLayerParam();
    static LayerParam generateLinearNormalizedLayerParam();
    static LayerParam generateNonLinearLayerParam();

    Layer() = default; // only used in NetworkIO::loadCoefs(). DO NOT USE MANUALLY
    Layer(LayerParam const& param);
    void init(size_t nbInputs, std::mt19937& generator);
    void init(size_t nbInputs);
    Matrix process(Matrix const& inputs, ThreadPool& t) const;
    Vector processToLearn(Vector const& input, std::bernoulli_distribution& dropoutDist, std::bernoulli_distribution& dropconnectDist, std::mt19937& dropGen, ThreadPool& t);
    Vector processToGenerate(Vector const& input, ThreadPool& t);
    void computeGradients(Vector const& inputGradient, ThreadPool& t);
    void computeGradientsAccordingToInputs(Vector const& inputGradient, ThreadPool& t);
    void keep();
    void release();
    Vector getGradients(ThreadPool& t); //one gradient per input neuron
    void updateWeights(double learningRate, double L1, double L2, double weightDecay, bool automaticLearningRate, bool adaptiveLearningRate, bool useMaxDenominator, double momentum, double previousMomentum, double nextMomentum, double cumulativeMomentum, double window, double optimizerBias, size_t iteration, ThreadPool& t);
    void updateInput(Vector& input, double learningRate);
    void resetGradientsForGeneration(ThreadPool& t);
    size_t size() const;
    void resize(size_t neurons);
    size_t inputSize() const;
    std::pair<double, double> L1L2(ThreadPool& t) const;
    size_t getNbParameters() const;

private:
    LayerParam _param;
    size_t _inputSize;
    std::vector<Neuron> _neurons;
};



void to_json(json& jObj, Layer const& layer);
void from_json(json const& jObj, Layer& layer);



} //namespace omnilearn



#endif //OMNILEARN_LAYER_HH_