//preprocess.cpp

#include "omnilearn/preprocess.hh"
#include "omnilearn/disable_eigen_warnings.h"
#include "omnilearn/Exception.hh"

DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
DISABLE_WARNING_EXTRA
DISABLE_WARNING_DEPRECATED_COPY
DISABLE_WARNING_OLD_STYLE_CAST
DISABLE_WARNING_CONVERSION
#include "omnilearn/eigen/SVD"
DISABLE_WARNING_POP



//subtract the mean of each comumn to each elements of these columns
//returns the mean of each column
omnilearn::Vector omnilearn::center(Matrix& data, Vector mean)
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
std::vector<std::pair<double, double>> omnilearn::normalize(Matrix& data, std::vector<std::pair<double, double>> mM)
{
  if(mM.size() == 0)
  {
    mM = std::vector<std::pair<double, double>>(data.cols(), {0, 0});
    //search min and max
    for(eigen_size_t i = 0; i < data.cols(); i++)
    {
      mM[i] = {data.col(i).minCoeff(), data.col(i).maxCoeff()};
      if(std::abs(mM[i].second - mM[i].first) < std::numeric_limits<double>::epsilon())
        mM[i].second++; // so the whole column is just set to 0
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
std::vector<std::pair<double, double>> omnilearn::standardize(Matrix& data, std::vector<std::pair<double, double>> meanDev)
{
  if(meanDev.size() == 0)
  {
    meanDev = std::vector<std::pair<double, double>>(data.cols(), {0, 0});
    //calculate mean and deviation
    for(eigen_size_t i = 0; i < data.cols(); i++)
    {
      meanDev[i] = {data.col(i).mean(), dev(data.col(i))};
      if(std::abs(meanDev[i].second) < std::numeric_limits<double>::epsilon())
        meanDev[i].second++; // so the whole column is just set to 0
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
std::pair<omnilearn::Matrix, omnilearn::Vector> omnilearn::decorrelate(Matrix& data, std::pair<Matrix, Vector> singular)
{
  if(singular.second.size() == 0)
  {
    Matrix cov = (data.transpose() * data) / static_cast<double>(data.rows() - 1);

    //in U, eigen vectors are columns
    Eigen::BDCSVD<Matrix> svd(cov, Eigen::ComputeFullU);
    singular.first = svd.matrixU();
    singular.second = svd.singularValues();
  }

  //apply rotation
  for(eigen_size_t i = 0; i < data.rows(); i++)
  {
    data.row(i) = singular.first.transpose() * data.row(i).transpose();
  }
  return singular;
}


void omnilearn::whiten(Matrix& data, std::pair<Matrix, Vector> const& singular, double bias)
{
  if(singular.second.size() == 0)
    throw Exception("Decorrelation must be performed before whitening");
  for(eigen_size_t i = 0; i < data.cols(); i++)
  {
    data.col(i) /= (std::sqrt(singular.second[i])+bias);
  }
}


void omnilearn::reduce(Matrix& data, std::pair<Matrix, Vector> const& singular, double threshold)
{
  if(singular.second.size() == 0)
    throw Exception("Decorrelation must be performed before reduction");

  double eigenTot = singular.second.sum();
  double eigenSum = 0;

  for(eigen_size_t i = 0; i < singular.second.size(); i++)
  {
    eigenSum += singular.second[i];
    if(eigenSum/eigenTot >= threshold)
    {
      data = Matrix(data.leftCols(i+1));
      break;
    }
  }
}