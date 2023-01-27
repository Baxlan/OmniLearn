// preprocess.h

#ifndef OMNILEARN_PREPROCESS_H_
#define OMNILEARN_PREPROCESS_H_

#include "Matrix.hh"
#include "Network.hh"



namespace omnilearn
{



std::vector<std::pair<double, double>> normalize(Matrix& data, std::vector<std::pair<double, double>> mM = {}); //returns vector of min and max respectively
std::vector<std::pair<double, double>> standardize(Matrix& data, std::vector<std::pair<double, double>> meanDev = {}); //returns vector of mean and deviation respectively
DecorrelationData whiten(Matrix& data, double bias, WhiteningType whiteningType, std::vector<std::string> const& infos, DecorrelationData decorrelationData = DecorrelationData());
void reduce(Matrix& data, DecorrelationData const& decorrelationData, double threshold);

void deNormalize(Matrix& data, std::vector<std::pair<double, double>> const& mM);
void deStandardize(Matrix& data, std::vector<std::pair<double, double>> const& meanDev);
void deWhiten(Matrix& data, double bias, WhiteningType whiteningType, DecorrelationData const& decorrelationData);
void deReduce(Matrix& data, DecorrelationData const& decorrelationData);



} //namespace omnilearn

#endif // OMNILEARN_PREPROCESS_H_