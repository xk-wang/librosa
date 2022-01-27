#ifndef DEFINE_H_
#define DEFINE_H_

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // !M_PI

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcf;

#endif