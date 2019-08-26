
#ifndef BRAIN_PRETREAT_HH_
#define BRAIN_PRETREAT_HH_

#include "utility.hh"

namespace brain
{


//subtract the mean of each comumn to each elements of these columns
//returns the mean of each column
std::vector<double> center(Matrix& data, std::vector<double> mean = {})
{
  if(mean.size() == 0)
  {
    mean = std::vector<double>(data[0].size(), 0);
    //calculate mean
    for(unsigned i = 0; i < data.size(); i++)
    {
      for(unsigned j = 0; j < data[0].size(); j++)
      {
        mean[j] += data[i][j];
      }
    }
    for(unsigned i = 0; i < mean.size(); i++)
    {
      mean[i] /= data.size();
    }
  }

  //center
  for(unsigned i = 0; i < data.size(); i++)
  {
    for(unsigned j = 0; j < data[0].size(); j++)
    {
      data[i][j] -= mean[j];
    }
  }
  return mean;
}


//set the elements of each columns between a range [0, 1]
//returns vector of min and max respectively
std::vector<std::pair<double, double>> normalize(Matrix& data, std::vector<std::pair<double, double>> minMax = {})
{
  if(minMax.size() == 0)
  {
    minMax = std::vector<std::pair<double, double>>(data[0].size(), {std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()});
    //search min and max
    for(unsigned i = 0; i < data.size(); i++)
    {
      for(unsigned j = 0; j < data[0].size(); j++)
      {
        if(data[i][j] < minMax[j].first)
        {
          minMax[j].first = data[i][j];
        }
        else if(data[i][j] > minMax[j].second)
        {
          minMax[j].second = data[i][j];
        }
      }
    }
  }
  //normalize
  for(unsigned i = 0; i < data.size(); i++)
  {
    for(unsigned j = 0; j < data[0].size(); j++)
    {
      data[i][j] = (data[i][j] - minMax[j].first) / (minMax[j].second - minMax[j].first);
    }
  }
  return minMax;
}


//set the mean and the std deviation of each column to 0 and 1
//returns vector of mean and deviation respectively
std::vector<std::pair<double, double>> standardize(Matrix& data, std::vector<std::pair<double, double>> meanDev = {})
{
  if(meanDev.size() == 0)
  {
    meanDev = std::vector<std::pair<double, double>>(data[0].size(), {0, 0});
    //calculate mean
    for(unsigned i = 0; i < data.size(); i++)
    {
      for(unsigned j = 0; j < data[0].size(); j++)
      {
        meanDev[j].first += data[i][j];
      }
    }
    for(unsigned i = 0; i < meanDev.size(); i++)
    {
      meanDev[i].first /= data.size();
    }
    //calculate deviation
    for(unsigned i = 0; i < data.size(); i++)
    {
      for(unsigned j = 0; j < data[0].size(); j++)
      {
        meanDev[j].second += std::pow(meanDev[j].first - data[i][j], 2);
      }
    }
    for(unsigned i = 0; i < meanDev.size(); i++)
    {
      meanDev[i].second /= data.size();
      meanDev[i].second = std::sqrt(meanDev[i].second);
    }
  }

  //standardize
  for(unsigned i = 0; i < data.size(); i++)
  {
    for(unsigned j = 0; j < data[0].size(); j++)
    {
      data[i][j] -= meanDev[j].first;
      data[i][j] /= meanDev[j].second;
    }
  }
  return meanDev;
}


//rotate data in the decorrelated space (and set their variance to 1).
//USE THIS FUNCTION ONLY IF DATA ARE MEAN CENTERED
//returns the eigenvectors of the cov matrix of the data
Matrix whiten(Matrix& data, Matrix eigen = {})
{

}


} //namespace brain

#endif // BRAIN_PRETREAT_HH_