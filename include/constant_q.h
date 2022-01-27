#ifndef CONSTANT_Q_H_
#define CONSTANT_Q_H_

#include <cmath>
#include <complex>
#include <string>
#include <algorithm>
#include <exception>

#include "audio.h"
#include "stft.h"
#include "util.h"
#include "filters.h"

namespace librosa{
  namespace cqt{

    static int __cqt_filter_fft(Matrixcf& fft_basis,
                          float sr,
                          float fmin, 
                          int n_bins, 
                          int bins_per_octave, 
                          float tuning,
                          float filter_scale, 
                          int norm, 
                          const std::string& window="hann",
                          int hop_length=-1){

      Matrixcf basis;
      Vectorf lengths;

      basis = librosa::filters::constant_q(lengths, sr, fmin, n_bins, bins_per_octave, 
                                   tuning, window, filter_scale, true, norm); 
                          
      int n_frames = basis.rows(), n_fft = basis.cols(), n_f = n_fft/2+1;
      if(hop_length>0 && n_fft < std::pow(2.f, 1+std::ceil(std::log2(hop_length)))){
        n_fft = int(std::pow(2.f, 1+std::ceil(std::log2f(hop_length))));
      }

      basis = basis.array() * (lengths.replicate(n_fft, 1).transpose().array()) / float(n_fft);

      Eigen::FFT<float> fft;
      Matrixcf out_basis(n_frames, n_fft);
      for (int i = 0; i < n_frames; ++i){
        out_basis.row(i) = fft.fwd(basis.row(i), n_fft);
      }
      fft_basis = out_basis.leftCols(n_f);

      return n_fft;
    }

    static Matrixcf __cqt_response(const Vectorf& y, 
                                int n_fft,
                                int hop_length,
                                const Matrixcf& fft_basis,
                                const std::string& mode){
      
      Matrixcf D = stft::stft(y, n_fft, hop_length, "ones", true, mode).transpose();
      return fft_basis*D;
    }

    static Vectorf cqt_frequencies(int n_bins, float f_min, int bins_per_octave=12, float tuning=0.0){

      float correction = std::pow(2.f, tuning/bins_per_octave);
      Vectorf frequencies = f_min*pow(2.f, Vectorf::LinSpaced(n_bins, 0, n_bins-1).array()/bins_per_octave);
      return correction*frequencies;
    
    }

    static int _num_two_factors(int x){
      if(x<=0) return 0;
      int num_twos = 0;
      while(x%2==0){
        num_twos+=1;
        x/=2;
      }
      return num_twos;
    }

    static int __early_downsample_count(float nyquist, float filter_cutoff, int hop_length, int n_octaves){
      int downsample_count1 = std::max(0, int(std::ceil(std::log2f(librosa::audio::BW_FASTEST*nyquist/filter_cutoff)))-2);
      int num_twos = _num_two_factors(hop_length);
      int downsample_count2 = std::max(0, num_twos-n_octaves+1);
      return std::min(downsample_count1, downsample_count2);
    }

    static void __early_downsample(Vectorf& y, float&sr, int& hop_length, const std::string& res_type,
                          int n_octaves, float nyquist, float filter_cutoff, bool scale){
      int downsample_count = __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves);
      if(downsample_count>0 && res_type=="kaiser_fast"){
        int downsample_factor = int(std::pow(2, downsample_count));
        hop_length /= downsample_factor;
        if(y.size()<downsample_factor){
          throw std::invalid_argument("Input signal y is too short");
        }
        float new_sr = sr / float(downsample_factor);

        y = librosa::audio::resample(y, sr, new_sr, res_type, true, true);

        if(!scale) y*=std::sqrt(downsample_factor);
        sr = new_sr;
      }
    }

    static Matrixcf cqt(Vectorf& y, 
                        float sr=22050, 
                        int hop_length=512, 
                        float fmin=-1.f, 
                        int n_bins=84, 
                        int bins_per_octave=12,
                        float tuning=0.0, 
                        float filter_scale=1.f, 
                        int norm=1,
                        std::string window="hann", 
                        bool scale=true, 
                        std::string pad_mode="reflect") {
    
      int n_octaves = int(std::ceil(float(n_bins) / bins_per_octave));
      int n_filters = std::min(bins_per_octave, n_bins);

      if(fmin<0) fmin = 32.703196; // C1 fmin = note_to_hz("C1");
      if(tuning<0){
        throw std::invalid_argument("estimate tuning is not supported!");
      }
      // if(tuning<0) estimate_tuning(y, sr);

      Vectorf freqs = cqt_frequencies(n_bins, fmin, bins_per_octave).rightCols(bins_per_octave);  
      float fmin_t = freqs.minCoeff();
      float fmax_t = freqs.maxCoeff();

      float Q = filter_scale / (std::pow(2.f, 1.f/bins_per_octave)-1);
      float filter_cutoff = fmax_t * (1+0.5*(filters::WINDOW_BANDWIDTHS[window])/Q);
      float nyquist = sr/2.f;
      std::string res_type;
      if(filter_cutoff < librosa::audio::BW_FASTEST*nyquist){
        res_type = "kaiser_fast";
      }else{
        res_type = "kaiser_best";
      }

      __early_downsample(y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale);

      int frames = y.size()/hop_length+1;
      
      Matrixcf cqt_resp(bins_per_octave*n_octaves, frames);

      Matrixcf fft_basis;
      int n_fft;

      int octave_offset = 0;
      if(res_type!="kaiser_fast"){

        n_fft = __cqt_filter_fft(fft_basis, sr, fmin_t, n_filters, bins_per_octave, tuning,
                                    filter_scale, norm, window);
        cqt_resp.bottomRows(n_filters) = __cqt_response(y, n_fft, hop_length, fft_basis, pad_mode).colwise().reverse();

        fmin_t /= 2;
        fmax_t /= 2;
        n_octaves -= 1;
        
        filter_cutoff = fmax_t * (1+0.5*filters::WINDOW_BANDWIDTHS[window]/Q);
        res_type = "kaiser_fast";
        octave_offset = 1;
      }

      int num_twos = _num_two_factors(hop_length);
      if(num_twos < n_octaves-1){
        throw std::invalid_argument("hop_length is too short");
      }

      n_fft = __cqt_filter_fft(fft_basis, sr, fmin_t, n_filters, bins_per_octave, tuning, 
                               filter_scale, norm, window);
      
      Vectorf my_y = y;

      
      float my_sr = sr;
      int my_hop = hop_length;

      for(int i=0; i<n_octaves; ++i){

        if(i>0){

          if(my_y.size()<2){
            throw std::invalid_argument("input signal is too short");
          }

          my_y = librosa::audio::resample(my_y, my_sr, my_sr / 2.f, res_type, true, true);

          fft_basis *= std::sqrt(2);
      // std::cout << "fft_basis: " <<" " << fft_basis.array().abs().rowwise().sum().transpose() << std::endl << std::endl;

          my_sr /= 2.f;
          my_hop /= 2;
        }
      
      // std::cout << "my_y: " << my_y.array().abs().sum() << " " << my_y.size() << std::endl;

      //  std::cout << "D " << __cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode).array().abs().sum() << std::endl;
      cqt_resp.block((n_octaves-i-1)*n_filters, 0, n_filters, frames) = __cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode).colwise().reverse();

      }

      Matrixcf C = cqt_resp.topRows(n_bins);
      std::cout << "C: " << C.array().abs().rowwise().sum().transpose() << std::endl;

      if(scale){
        Vectorf lengths = filters::constant_q_lengths(sr, fmin, n_bins, bins_per_octave, tuning, window, filter_scale);
        C = C.array() / lengths.replicate(C.cols(), 1).transpose().array().sqrt();
      }

      return C;
    }

    static Matrixf cqtspectrogram(Vectorf& y,
                                 float sr=22050, 
                                 int hop_length=512, 
                                 float fmin=-1.f, 
                                 int n_bins=84, 
                                 int bins_per_octave=12,
                                 float tuning=0.0, 
                                 float filter_scale=1.f, 
                                 int norm=1,
                                 std::string window="hann", 
                                 bool scale=true, 
                                 std::string pad_mode="reflect",
                                 float power=1.f){


      Matrixcf C = cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale,
              norm, window, scale, pad_mode);
              
      Matrixf spectrogram = librosa::util::spectrogram(C, power);
      return spectrogram;    
    }

  }
}

#endif