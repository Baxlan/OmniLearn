
#ifndef BRAIN_PREPROCESS_HH_
#define BRAIN_PREPROCESS_HH_

#include "Matrix.hh"

namespace brain
{


//subtract the mean of each comumn to each elements of these columns
//returns the mean of each column
Vector center(Matrix& data, Vector mean = Vector(0))
{
  if(mean.size() == 0)
  {
    mean = Vector(data.cols());
    //calculate mean
    for(eigen_size_t i = 0; i < data.cols(); i++)
    {
      mean[i] = data.col(i).mean();
    }
  }

  //center
  for(eigen_size_t i = 0; i < data.rows(); i++)
  {
    for(eigen_size_t j = 0; j < data.cols(); j++)
    {
      data(i, j) -= mean[j];
    }
  }
  return mean;
}


//set the elements of each columns between a range [0, 1]
//returns vector of min and max respectively
std::vector<std::pair<double, double>> normalize(Matrix& data, std::vector<std::pair<double, double>> mM = {})
{
  if(mM.size() == 0)
  {
    mM = std::vector<std::pair<double, double>>(data.cols(), {0, 0});
    //search min and max
    for(eigen_size_t i = 0; i < data.cols(); i++)
    {
      mM[i] = {data.col(i).minCoeff(), data.col(i).maxCoeff()};
      //if all values of a column are the same, divide by 0.
      //this in this case, we divide by the data itself to normalize to 1
      if(std::abs(mM[i].second - mM[i].first) < std::numeric_limits<double>::epsilon())
        mM[i] = {0, data(0, i)};
    }
  }
  //normalize
  for(eigen_size_t i = 0; i < data.rows(); i++)
  {
    for(eigen_size_t j = 0; j < data.cols(); j++)
    {
      data(i, j) = (data(i, j) - mM[j].first) / (mM[j].second - mM[j].first);
    }
  }
  return mM;
}


//set the mean and the std deviation of each column to 0 and 1
//returns vector of mean and deviation respectively
std::vector<std::pair<double, double>> standardize(Matrix& data, std::vector<std::pair<double, double>> meanDev = {})
{
  if(meanDev.size() == 0)
  {
    meanDev = std::vector<std::pair<double, double>>(data.cols(), {0, 0});
    //calculate mean and deviation
    for(eigen_size_t i = 0; i < data.cols(); i++)
    {
      meanDev[i] = {data.col(i).mean(), dev(data.col(i))};
      //if all values of a column are the same, divide by 0 (dev is 0).
      //this in this case, we divide by 1
      if(std::abs(meanDev[i].second) < std::numeric_limits<double>::epsilon())
        meanDev[i].second = 1;
    }
  }
  //standardize
  for(eigen_size_t i = 0; i < data.rows(); i++)
  {
    for(eigen_size_t j = 0; j < data.cols(); j++)
    {
      data(i, j) -= meanDev[j].first;
      data(i, j) /= meanDev[j].second;
    }
  }
  return meanDev;
}


//rotate data in the input space to decorrelate them (and set their variance to 1).
//USE THIS FUNCTION ONLY IF DATA ARE MEAN CENTERED
//first is rotation matrix (eigenvectors of the cov matrix of the data), second is eigenvalues
std::pair<Matrix, Vector> whiten(Matrix& data, Matrix rotation = Matrix(0,0))
{

}



} //namespace brain

#endif // BRAIN_PREPROCESS_HH_