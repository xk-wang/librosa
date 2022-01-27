#ifndef MEL_H_
#define MEL_H_

#include <cmath>
#include <string>

#include "util.h"
#include "stft.h"
#include "filters.h"

namespace librosa{
  namespace mel{

    static Matrixf melspectrogram(Vectorf &x, int sr, int n_fft, int n_hop,
                          const std::string &win, bool center,
                          const std::string &mode, float power,
                          int n_mels, int fmin, int fmax, bool htk = false){
      Matrixcf X = librosa::stft::stft(x, n_fft, n_hop, win, center, mode);
      Matrixf mel_basis = librosa::filters::melfilter(sr, n_fft, n_mels, fmin, fmax, htk);
      Matrixf sp = librosa::util::spectrogram(X, power);
      Matrixf mel = mel_basis*sp.transpose();
      return mel.transpose();
    }

    static Matrixf dct(Matrixf& x, bool norm, int type) {
      int N = x.cols();
      Matrixf xi = Matrixf::Zero(N, N);
      xi.rowwise() += Vectorf::LinSpaced(N, 0.f, static_cast<float>(N-1));
      // type 2
      Matrixf coeff = 2*(M_PI*xi.transpose().array()/N*(xi.array()+0.5)).cos();
      Matrixf dct = x*coeff.transpose();
      // ortho
      if (norm) {
        Vectorf ortho = Vectorf::Constant(N, sqrtf(0.5f/N));
        ortho[0] = std::sqrt(0.25f/N);
        dct = dct*ortho.asDiagonal();
      }
      return dct;
    }

  }
}

#endif