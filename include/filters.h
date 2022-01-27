#ifndef FILTERS_H_
#define FILTERS_H_

#include <complex>
#include <unordered_map>
#include <string>
#include <exception>
#include <cmath>
#include <algorithm>

#include "util.h"

namespace librosa{
  namespace filters{

    static std::unordered_map<std::string, double> WINDOW_BANDWIDTHS = {
      {"bart", 1.333496133},
      {"barthann", 1.456025597},
      {"bartlett", 1.333496133},
      {"bkh", 2.004597528},
      {"black", 1.726968155},
      {"blackharr", 2.004597528},
      {"blackman", 1.726968155},
      {"blackmanharris", 2.004597528},
      {"blk", 1.726968155},
      {"bman", 1.785958861},
      {"bmn", 1.785958861},
      {"bohman", 1.785958861},
      {"box", 1.0},
      {"boxcar", 1.0},
      {"brt", 1.333496133},
      {"brthan", 1.456025597},
      {"bth", 1.456025597},
      {"cosine", 1.233700535},
      {"flat", 2.776225505},
      {"flattop", 2.776225505},
      {"flt", 2.776225505},
      {"halfcosine", 1.233700535},
      {"ham", 1.362945532},
      {"hamm", 1.362945532},
      {"hamming", 1.362945532},
      {"han", 1.500183105},
      {"hann", 1.500183105},
      {"hanning", 1.500183105},
      {"nut", 1.976350028},
      {"nutl", 1.976350028},
      {"nuttall", 1.976350028},
      {"ones", 1.0},
      {"par", 1.917460317},
      {"parz", 1.917460317},
      {"parzen", 1.917460317},
      {"rect", 1.0},
      {"rectangular", 1.0},
      {"tri", 1.333170652},
      {"triang", 1.333170652},
      {"triangle", 1.333170652}
    };

    static Vectorf get_window(const std::string& win, int n_fft){
      // if(fft_bins)
      Vectorf window;
      if(win=="hann")
      {
        window= 0.5*(1.f-(Vectorf::LinSpaced(n_fft, 0.f, static_cast<float>(n_fft-1))*2.f*M_PI/n_fft).array().cos());
      }
      else if(win=="ones")
      {
        window = Vectorf::Ones(n_fft);
      }
      return window;
    }

    static Vectorf constant_q_lengths(float sr, float fmin, int n_bins=84, int bins_per_octave=12,
                                    float tuning=0.0, const std::string& window="hann", 
                                    float filter_scale=1.f){
      if(fmin<0){
        throw std::invalid_argument("fmin must be positive");
      }
      if(bins_per_octave<=0){
        throw std::invalid_argument("bins_per_octave must be positive");
      }
      if(filter_scale<=0){
        throw std::invalid_argument("filter_scale must be positive");
      }
      if(n_bins<=0){
        throw std::invalid_argument("n_bins must be positive");
      }

      float correction = std::pow(2.f, tuning / bins_per_octave);
      fmin = correction*fmin;

      float Q = filter_scale / (std::pow(2.f, 1.f/bins_per_octave)-1);

      Vectorf coeff = Vectorf::LinSpaced(n_bins, 0.f, static_cast<float>(n_bins-1));
      coeff /= bins_per_octave;
      Vectorf freq = fmin*pow(2.f, coeff.array());

      // std::cout << "max_freq: " << freq(n_bins-1) << std::endl
      //           << "nyquist: " << sr/2.f << std::endl
      //           << "bins_per_octave: " << bins_per_octave << std::endl
      //           << freq << std::endl
      //           << coeff <<std::endl
      //           << pow(2.f, coeff.array()) << std::endl
      //           << "fmin: " << fmin << std::endl;

      if(freq(n_bins-1)*(1+0.5*WINDOW_BANDWIDTHS[window]/Q) > sr/2.f){
        throw std::invalid_argument("Filter pass-band lies beyond Nyquist");
      }
      Vectorf lengths = Q*sr/freq.array();
      return lengths;
    }

    static Matrixcf constant_q(Vectorf& lengths,
                           float sr, float fmin=-1, int n_bins=84, 
                           int bins_per_octave=12, float tuning=0.0,
                           const std::string& window="hann", float filter_scale=1.f,
                           bool pad_fft=true, int norm=1){
      
      if(fmin<0) fmin = 32.703196; // C1

      lengths = constant_q_lengths(sr, fmin, n_bins, bins_per_octave, tuning, window, filter_scale);

      float correction = std::pow(2.f, tuning / bins_per_octave);
      fmin = correction*fmin;
      float Q = filter_scale / (std::pow(2.f, 1.f/bins_per_octave)-1);
      auto freqs = Q*sr/lengths.array();

      int max_len;
      if(pad_fft) max_len = int(std::pow(2.f, std::ceil(std::log2f(lengths.maxCoeff()))));
      else max_len = int(std::ceil(lengths.maxCoeff()));
      Matrixcf filters(lengths.size(), max_len);

      std::complex<float> a(0, 1);

      for(int i=0; i<lengths.size(); ++i){
        int ilen = int(lengths[i]);
        float freq = freqs[i];
        Vectorcf sig = exp(Vectorf::LinSpaced(ilen, static_cast<float>(std::floor(-ilen/2.f)), static_cast<float>(ilen/2-1))
                      .array()*a*2*M_PI*freq/sr);     
        
        // 计算窗函数
        sig = sig.array()*(get_window(window, sig.size()).array());
        
        // sig = normalize(sig, norm=norm); 完成normalize的步骤 按照行相加归一化
        float norm = sig.lpNorm<1>();

        if(norm > 1e-7){
          sig /= norm;
        }
        
        int left = (max_len - ilen)/2, right = max_len - ilen - left;
        filters.row(i) = util::pad(sig, left, right, "constant");
      }

      return filters;
    }

    static Matrixf melfilter(int sr, int n_fft, int n_mels, int fmin, int fmax, bool htk = false){
      int n_f = n_fft/2+1;
      Vectorf fft_freqs = (Vectorf::LinSpaced(n_f, 0.f, static_cast<float>(n_f-1))*sr)/n_fft;

      float f_min = 0.f;
      float f_sp = 200.f/3.f;
      float min_log_hz = 1000.f;
      float min_log_mel = (min_log_hz-f_min)/f_sp;
      float logstep = logf(6.4f)/27.f;

      auto hz_to_mel = [=](int hz, bool htk) -> float {
        if (htk){
          return 2595.0f*log10f(1.0f+hz/700.0f);
        }
        // 另一种方式的mel频点
        float mel = (hz-f_min)/f_sp;
        if (hz >= min_log_hz){
          mel = min_log_mel+logf(hz/min_log_hz)/logstep;
        }
        return mel;
      };
      auto mel_to_hz = [=](Vectorf &mels, bool htk) -> Vectorf {
        if (htk){
          return 700.0f*(Vectorf::Constant(n_mels+2, 10.f).array().pow(mels.array()/2595.0f)-1.0f);
        }
        return (mels.array()>min_log_mel).select(((mels.array()-min_log_mel)*logstep).exp()*min_log_hz, (mels*f_sp).array()+f_min);
      };

      float min_mel = hz_to_mel(fmin, htk);
      float max_mel = hz_to_mel(fmax, htk);
      Vectorf mels = Vectorf::LinSpaced(n_mels+2, min_mel, max_mel);
      Vectorf mel_f = mel_to_hz(mels, htk);
      Vectorf fdiff = mel_f.segment(1, mel_f.size() - 1) - mel_f.segment(0, mel_f.size() - 1);
      // replicate(rowfactor, colfactor) 将行和列复制的次数
      Matrixf ramps = mel_f.replicate(n_f, 1).transpose().array() - fft_freqs.replicate(n_mels + 2, 1).array();

      Matrixf lower = -ramps.topRows(n_mels).array()/fdiff.segment(0, n_mels).transpose().replicate(1, n_f).array();
      Matrixf upper = ramps.bottomRows(n_mels).array()/fdiff.segment(1, n_mels).transpose().replicate(1, n_f).array();
      // select函数满足条件选前者，否则选后者 cwiseMax逐个元素选最大值，后面的元素可以是矩阵也可是标量
      Matrixf weights = (lower.array()<upper.array()).select(lower, upper).cwiseMax(0);

      auto enorm = (2.f/(mel_f.segment(2, n_mels)-mel_f.segment(0, n_mels)).array()).transpose().replicate(1, n_f);
      weights = weights.array()*enorm;

      return weights;
    }
  }
}

#endif