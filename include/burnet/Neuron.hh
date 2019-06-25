#ifndef BURNET_NEURON_HH_
#define BURNET_NEURON_HH_

#include "Activation.hh"
#include "Aggregation.hh"

#include <memory>

namespace burnet
{



class Neuron
{
public:
    double process(std::vector<double> const& inputs)
    {
        return _Activation->activate(_Aggregation->aggregate(inputs).first);
    }


    double processToLearn(std::vector<double> const& inputs)
    {
        if(_currentFeature >= (_batchSize - 1))
        {
            throw Exception("Batch size have been reashed, gradients must be calculated then parameters must be updated before processing new features.");
        }
        _currentFeature++;
        _aggregResults[_currentFeature-1] = _Aggregation->aggregate(inputs);
        _activResults[_currentFeature-1] = _Activation->activate(_aggregResults[_currentFeature-1].first);
        return _activResults[_currentFeature-1];
    }


    void computeGradient(std::vector<double> inputGradients) //one input by feature in the batch
    {
        if(_currentFeature < (_batchSize - 1))
        {
            throw Exception("Calculating gradient but the batch size have not been reashed.");
        }
        else if(_currentFeature > (_batchSize - 1))
        {
            throw Exception("Calculating gradient but parameters have not be updated.");
        }

        std::vector<double> actGradients(_batchSize, 0); //store gradients between activation and aggregation for each feature of the batch

        for(_currentFeature = 0; _currentFeature < _batchSize; _currentFeature++)
        {
            _inputGradients[_currentFeature] = inputGradients[_currentFeature];
            actGradients[_currentFeature] = _inputGradients[_currentFeature] * _Activation->prime(_activResults[_currentFeature]); // * aggreg.prime ?
            _averageActGradient = average(actGradients);
        }

        //setting all partial gradients on 0
        _gradients = std::vector<std::vector<double>>(_Aggregation->k(), std::vector<double>(_batchSize, 0));

        //storing new partial gradients
        for(_currentFeature = 0; _currentFeature < _currentFeature; _currentFeature++)
        {
            _gradients[_aggregResults[_currentFeature].second][_currentFeature] = _aggregResults[_currentFeature].first * actGradients[_currentFeature];
        }

        _currentFeature++;
    }


    void updateWeights(double learningRate, double L1, double L2, double tackOn, double maxNorm, double momentum)
    {
        if(_currentFeature <= _batchSize - 1)
        {
            throw Exception("Updating parameters but gradients have not been calculated.");
        }

        _Activation->learn(average(_inputGradients), 0, 0);
        _Aggregation->learn(_averageActGradient, 0, 0);

        //for each weight set
        for(unsigned i = 0; i < _Aggregation->k(); i++)
        {
            std::pair<std::vector<double>&, double&> w = _Aggregation->weightRef(i);
            double gradient = average(_gradients[i]);

            for(unsigned j = 0; j < w.first.size(); j++)
            {
                w.first[j] += (learningRate*(gradient + (L2*w.first[j]) + L1) + tackOn);
            }
            w.second += learningRate * gradient; // to divide by inputs
        }
    }

protected:
    std::shared_ptr<Activation> _Activation;   //should be a value but polymorphism is needed
    std::shared_ptr<Aggregation> _Aggregation; //should be a value but polymorphism is needed

    unsigned const _batchSize;

    unsigned _currentFeature; // current feature number in mini-batch
    std::vector<std::pair<double, unsigned>> _aggregResults; // results obtained by aggregation and weight set used, for each feature in the batch
    std::vector<double> _activResults; // results obtained by activation, for each feature in the batch

    std::vector<std::vector<double>> _previousWeightUpdate; // previous update aplied to each weight in each weight set
    std::vector<double> _previousBiasUpdate; // previous update aplied to bias for each weight set
    std::vector<double> _inputGradients; //for each feature of the batch, gradients entered from previous layers
    double _averageActGradient; //averaged gradient over all features of the batch, between activation and aggregation
    std::vector<std::vector<double>> _gradients; // partial gradient obtained for each feature of the batch and for each weight set
    //weight set //feature //partial gradient
};



} //namespace burnet



#endif //BURNET_NEURON_HH_