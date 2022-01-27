#ifndef UTIL_H_
#define UTIL_H_

#include <string>

#include "define.h"

namespace librosa{
  namespace util{

    template <typename T>
    static T pad(const T &x, int left, int right, const std::string &mode, float value=0.f){
      T x_paded = T::Constant(left+x.size()+right, value);
      x_paded.segment(left, x.size()) = x;

      if (mode.compare("reflect") == 0){
        for (int i = 0; i < left; ++i){
          x_paded[i] = x[left-i];
        }
        for (int i = left; i < left+right; ++i){
          x_paded[i+x.size()] = x[x.size()-2-i+left];
        }
      }

      if (mode.compare("symmetric") == 0){
        for (int i = 0; i < left; ++i){
          x_paded[i] = x[left-i-1];
        }
        for (int i = left; i < left+right; ++i){
          x_paded[i+x.size()] = x[x.size()-1-i+left];
        }
      }

      if (mode.compare("edge") == 0){
        for (int i = 0; i < left; ++i){
          x_paded[i] = x[0];
        }
        for (int i = left; i < left+right; ++i){
          x_paded[i+x.size()] = x[x.size()-1];
        }
      }

      if (mode.compare("constant") == 0){
        for (int i = 0; i < left; ++i){
          x_paded[i] = 0;
        }
        for (int i = left; i < left+right; ++i){
          x_paded[i+x.size()] = 0;
        }
      }

      return x_paded;
    }

    static Matrixf power2db(Matrixf& x) {
      auto log_sp = 10.0f*x.array().max(1e-10).log10();
      return log_sp.cwiseMax(log_sp.maxCoeff() - 80.0f);
    }

    static Vectorf fix_length(const Vectorf&y, int size){
      int n = y.size();
      Vectorf new_y;
      if(n>size){
        new_y = y.rightCols(size);
      }
      else if(n<size)
      {
        new_y = pad(y, 0, size - n, "constant");
      }
      else{
        new_y = y;
      }
      return new_y;
    }

    static Matrixf spectrogram(Matrixcf &X, float power = 1.f){
      // cwiseAbs求每个元素的模 pow 幂
      return X.cwiseAbs().array().pow(power);
    }

  }
}

#endif