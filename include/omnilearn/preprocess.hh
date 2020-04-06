// preprocess.hh

#ifndef OMNILEARN_PREPROCESS_HH_
#define OMNILEARN_PREPROCESS_HH_

#include "Exception.hh"
#include "Matrix.hh"

#include "eigen/SVD"



namespace omnilearn
{



//subtract the mean of each comumn to each elements of these columns
//returns the mean of each column
Vector center(Matrix& data, Vector mean = Vector(0));

//set the elements of each columns between a range [0, 1]
//returns vector of min and max respectively
std::vector<std::pair<double, double>> normalize(Matrix& data, std::vector<std::pair<double, double>> mM = {});

//set the mean and the std deviation of each column to 0 and 1
//returns vector of mean and deviation respectively
std::vector<std::pair<double, double>> standardize(Matrix& data, std::vector<std::pair<double, double>> meanDev = {});

//rotate data in the input space to decorrelate them (and set their variance to 1).
//USE THIS FUNCTION ONLY IF DATA ARE MEAN CENTERED
//first is rotation matrix (eigenvectors of the cov matrix of the data), second is eigenvalues
std::pair<Matrix, Vector> decorrelate(Matrix& data, std::pair<Matrix, Vector> singular = {Matrix(0,0), Vector(0)});
void whiten(Matrix& data, std::pair<Matrix, Vector> const& singular, double bias);
void reduce(Matrix& data, std::pair<Matrix, Vector> const& singular, double threshold);



} //namespace omnilearn

#endif // OMNILEARN_PREPROCESS_HH_