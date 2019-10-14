
#ifndef BRAIN_PREPROCESS_HH_
#define BRAIN_PREPROCESS_HH_

#include "Matrix.hh"

namespace brain
{


//subtract the mean of each comumn to each elements of these columns
//returns the mean of each column
Vector center(Matrix& data, Vector mean = {})
{
  if(mean.size() == 0)
  {
    mean = Vector(data[0].size(), 0);
    //calculate mean
    for(unsigned i = 0; i < data[0].size(); i++)
    {
      mean[i] = data.column(i).mean().first;
    }
  }

  //center
  for(unsigned i = 0; i < data.lines(); i++)
  {
    for(unsigned j = 0; j < data.columns(); j++)
    {
      data[i][j] -= mean[j];
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
    mM = std::vector<std::pair<double, double>>(data[0].size(), {0, 0});
    //search min and max
    for(unsigned i = 0; i < data[0].size(); i++)
    {
      mM[i] = data.column(i).minMax();
    }
  }
  //normalize
  for(unsigned i = 0; i < data.lines(); i++)
  {
    for(unsigned j = 0; j < data.columns(); j++)
    {
      data[i][j] = (data[i][j] - mM[j].first) / (mM[j].second - mM[j].first);
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
    meanDev = std::vector<std::pair<double, double>>(data[0].size(), {0, 0});
    //calculate mean and deviation
    for(unsigned i = 0; i < data[0].size(); i++)
    {
      meanDev[i] = data.column(i).mean();
    }
  }
  //standardize
  for(unsigned i = 0; i < data.lines(); i++)
  {
    for(unsigned j = 0; j < data.columns(); j++)
    {
      data[i][j] -= meanDev[j].first;
      data[i][j] /= meanDev[j].second;
    }
  }
  return meanDev;
}


//rotate data in the input space to decorrelate them (and set their variance to 1).
//USE THIS FUNCTION ONLY IF DATA ARE MEAN CENTERED
//first is rotation matrix (eigenvectors of the cov matrix of the data), second is eigenvalues
std::pair<Matrix, Vector> whiten(Matrix& data, Matrix rotation = {})
{

}



} //namespace brain

#endif // BRAIN_PREPROCESS_HH_