#ifndef STFT_H_
#define STFT_H_

#include <string>

#include "util.h"
#include "filters.h"

namespace librosa{
  namespace stft{
    
    static Matrixcf stft(const Vectorf &x, int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode){
      // hanning
      Vectorf window = librosa::filters::get_window(win, n_fft);

      int pad_len = center ? n_fft / 2 : 0;
      Vectorf x_paded = librosa::util::pad(x, pad_len, pad_len, mode, 0.f);

      int n_f = n_fft/2+1;
      int n_frames = 1+(x_paded.size()-n_fft) / n_hop;
      Matrixcf X(n_frames, n_fft);
      Eigen::FFT<float> fft;

      for (int i = 0; i < n_frames; ++i){
        // segment(pos, n)
        Vectorf x_frame = window.array()*x_paded.segment(i*n_hop, n_fft).array();
        X.row(i) = fft.fwd(x_frame);
      }
      return X.leftCols(n_f);
    }

  }
}

#endif