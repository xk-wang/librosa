#include <eigen3/Eigen/Core>
#include <iostream>
// #include "myaudio.h"
// #include "filters.h"

#define pow2(x) std::pow(2.f, x)

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor> Arrayf;

template <typename T>
void func(T a, int b){
    std::cout << a << b << std::endl;
}

int main(int argc, char* argv[]){
  func(1, 2);
  Vectorf x=Vectorf::LinSpaced(10, 0, 9);
  Vectorf y=Vectorf::LinSpaced(5, 0, 4);
  std::cout << x << std::endl;
  std::cout << y.size() << " " << y << std::endl;
  y = x;
  std::cout << y.size() << " " << y << std::endl;
  return 0;
}