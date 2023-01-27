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



//=============================================================================
//=============================================================================
//=============================================================================
//======== PREPROCESSING ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



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


omnilearn::DecorrelationData omnilearn::whiten(Matrix& data, double bias, WhiteningType whiteningType, std::vector<std::string> const& infos, DecorrelationData decorrelationData)
{
  if(decorrelationData.eigenValues.size() == 0)
  {
    decorrelationData.dummyScales = Vector::Constant(data.cols(), 0);
    decorrelationData.dummyMeans = Vector::Constant(data.cols(), 0);

    // prepare for FAMD
    for(size_t i = 0; i < infos.size(); i++)
    {
      if(infos[i].find("dummy") != std::string::npos)
      {
          decorrelationData.dummyMeans[i] = data.col(i).mean();

          if(decorrelationData.dummyMeans[i] < std::numeric_limits<double>::epsilon())
            decorrelationData.dummyScales[i] = 1; // avoid dividing by 0
          else
            decorrelationData.dummyScales[i] = 1 / std::sqrt(decorrelationData.dummyMeans[i]);

          data.col(i) = (data.col(i).array() * decorrelationData.dummyScales[i]) - decorrelationData.dummyMeans[i];
      }
    }

    Matrix cov = (data.transpose() * data) / static_cast<double>(data.rows());

    //in U, eigen vectors are columns
    Eigen::BDCSVD<Matrix> svd(cov, Eigen::ComputeFullU);
    decorrelationData.eigenVectors = svd.matrixU();
    decorrelationData.eigenValues = svd.singularValues();
  }

  data = data * decorrelationData.eigenVectors * DiagMatrix((decorrelationData.eigenValues.cwiseSqrt().array() + bias).matrix().cwiseInverse());

  if(whiteningType == WhiteningType::ZCA)
  {
    data = data * decorrelationData.eigenVectors.transpose();
  }

  return decorrelationData;
}


void omnilearn::reduce(Matrix& data, DecorrelationData const& decorrelationData, double threshold)
{
  if(decorrelationData.eigenValues.size() == 0)
    throw Exception("Decorrelation must be performed before reduction");

  double eigenTot = decorrelationData.eigenValues.sum();
  double eigenSum = 0;

  for(eigen_size_t i = 0; i < decorrelationData.eigenValues.size(); i++)
  {
    eigenSum += decorrelationData.eigenValues[i];
    if(eigenSum/eigenTot >= threshold)
    {
      data = Matrix(data.leftCols(i+1));
      break;
    }
  }
}



//=============================================================================
//=============================================================================
//=============================================================================
//======== PREPROCESSING ======================================================
//=============================================================================
//=============================================================================
//=============================================================================



void omnilearn::deNormalize(Matrix& data, std::vector<std::pair<double, double>> const& mM)
{
  for(eigen_size_t i = 0; i < data.rows(); i++)
  {
    for(eigen_size_t j = 0; j < data.cols(); j++)
    {
      data(i,j) *= (mM[j].second - mM[j].first);
      data(i,j) += mM[j].first;
    }
  }
}


void omnilearn::deStandardize(Matrix& data, std::vector<std::pair<double, double>> const& meanDev)
{
  for(eigen_size_t i = 0; i < data.rows(); i++)
  {
    for(eigen_size_t j = 0; j < data.cols(); j++)
    {
      data(i,j) *= meanDev[j].second;
      data(i,j) += meanDev[j].first;
    }
  }
}


void omnilearn::deWhiten(Matrix& data, double bias, WhiteningType whiteningType, DecorrelationData const& decorrelationData)
{
  if(whiteningType == WhiteningType::ZCA)
  {
    data = data * decorrelationData.eigenVectors;
  }
  data = data * DiagMatrix((decorrelationData.eigenValues.cwiseSqrt().array() + bias).matrix()) * decorrelationData.eigenVectors.transpose();

  // reverse FAMD
  for(eigen_size_t i = 0; i < decorrelationData.dummyMeans.size(); i++)
  {
    data.col(i) = (data.col(i).array() + decorrelationData.dummyMeans[i]) / decorrelationData.dummyScales[i];
  }
}


void omnilearn::deReduce(Matrix& data, DecorrelationData const& decorrelationData)
{
  Matrix newResults(data.rows(), decorrelationData.eigenValues.size());
  rowVector zero = rowVector::Constant(decorrelationData.eigenValues.size() - data.cols(), 0);
  for(eigen_size_t i = 0; i < data.rows(); i++)
  {
    newResults.row(i) = (rowVector(decorrelationData.eigenValues.size()) << data.row(i), zero).finished();
  }
  data = newResults;
}