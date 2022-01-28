#ifndef DEFINE_H_
#define DEFINE_H_

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // !M_PI

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> Vectord;
typedef Eigen::Array<int, 1, Eigen::Dynamic, Eigen::RowMajor> Arrayi;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcd;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixd;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcf;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcd;


#endif