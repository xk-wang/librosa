#ifndef LIBROSA_H_
#define LIBROSA_H_
// #define EIGEN_USE_MKL_ALL
// #define EIGEN_VECTORIZE_SSE4_2

#include <string>
#include <vector>
#include <complex>

#include "util.h"
#include "audio.h"
#include "stft.h"
#include "mel.h"
#include "constant_q.h"

///
/// \brief c++ implemention of librosa
///
namespace librosa{

/// \brief      Load an audio file as a floating point time series
/// \param      path          path to the input file
/// \param      sr            target sampling rate
/// \param      mono          mono or dual
/// \param      offset        the begin pos of the audio
/// \param      duration      the length of the part want to get
/// \param      dtype         the data format of the return audio
/// \param      res_type      the filter type for resampling

static std::vector<float> load(const std::string& path,
                               int sr=22050,
                               bool mono=true,
                               float offset=0.f,
                               float duration=-1.f,
                               const std::string& dtype="float",
                               const std::string& res_type="kaiser_best"){

  return librosa::audio::load(path, sr, mono, offset, duration, dtype, res_type);
}


// 接口转换
class Feature
{
public:
  /// \brief      short-time fourier transform similar with librosa.feature.stft
  /// \param      x             input audio signal
  /// \param      n_fft         length of the FFT size
  /// \param      n_hop         number of samples between successive frames
  /// \param      win           window function. currently only supports 'hann' and 'ones'
  /// \param      center        same as librosa
  /// \param      mode          pad mode. support "reflect","symmetric","edge"
  /// \return     complex-valued matrix of short-time fourier transform coefficients.
  static std::vector<std::vector<std::complex<float>>> stft(std::vector<float> &x,
                                                            int n_fft, int n_hop,
                                                            const std::string &win, bool center,
                                                            const std::string &mode){
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixcf X = librosa::stft::stft(map_x, n_fft, n_hop, win, center, mode);
    std::vector<std::vector<std::complex<float>>> X_vector(X.rows(), std::vector<std::complex<float>>(X.cols(), 0));
    for (int i = 0; i < X.rows(); ++i){
      auto &row = X_vector[i];
      Eigen::Map<Vectorcf>(row.data(), row.size()) = X.row(i);
    }
    return X_vector;
  }

  /// \brief      constant Q transform similar with librosa.feature.cqt
  /// \param      x               input audio signal
  /// \param      hop_length      number of samples between successive frames
  /// \param      fmin            the low frequency
  /// \param      n_bins          n_bins of cqt
  /// \param      bins_per_octave bins_per_octave of cqt 
  /// \param      tuning          tuning of cqt
  /// \param      filter_scale    filter_scale of cqt
  /// \param      norm            norm of cqt       
  /// \param      window          window function. currently only supports 'hann' and 'ones'
  /// \param      scale           
  /// \param      pad_mode        pad mode. support "reflect","symmetric","edge"
  /// \return     complex-valued matrix of const Q transform coefficients.
  static std::vector<std::vector<std::complex<float>>> cqt(std::vector<float> &x,
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
                                                           std::string pad_mode="reflect"){
  
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixcf X = librosa::cqt::cqt(map_x, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm,
                               window, scale, pad_mode);
    std::vector<std::vector<std::complex<float>>> X_vector(X.rows(), std::vector<std::complex<float>>(X.cols(), 0));
    for (int i = 0; i < X.rows(); ++i){
      auto &row = X_vector[i];
      Eigen::Map<Vectorcf>(row.data(), row.size()) = X.row(i);
    }
    return X_vector;
  }

  static std::vector<std::vector<float>> abs_cqt(std::vector<float> &x,
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
                                                 std::string pad_mode="reflect"){
    
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixf X = librosa::cqt::cqt(map_x, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm,
                                  window, scale, pad_mode).array().abs();
    std::vector<std::vector<float>> X_vector(X.rows(), std::vector<float>(X.cols(), 0));
    for (int i = 0; i < X.rows(); ++i){
      auto &row = X_vector[i];
      Eigen::Map<Vectorf>(row.data(), row.size()) = X.row(i);
    }
    return X_vector;
  }

  /// \brief      compute mel spectrogram similar with librosa.feature.melspectrogram
  /// \param      x             input audio signal
  /// \param      sr            sample rate of 'x'
  /// \param      n_fft         length of the FFT size
  /// \param      n_hop         number of samples between successive frames
  /// \param      win           window function. currently only supports 'hann'
  /// \param      center        same as librosa
  /// \param      mode          pad mode. support "reflect","symmetric","edge"
  /// \param      power         exponent for the magnitude melspectrogram
  /// \param      n_mels        number of mel bands
  /// \param      f_min         lowest frequency (in Hz)
  /// \param      f_max         highest frequency (in Hz)
  /// \param      htk           using  htk frequency or not(boolean, default false)
  /// \return     mel spectrogram matrix
  static std::vector<std::vector<float>> melspectrogram(std::vector<float> &x, int sr, 
                                                        int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
                                                        float power, int n_mels, int fmin, int fmax, bool htk=false){
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixf mel = librosa::mel::melspectrogram(map_x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax, htk).transpose();
    std::vector<std::vector<float>> mel_vector(mel.rows(), std::vector<float>(mel.cols(), 0.f));
    for (int i = 0; i < mel.rows(); ++i){
      auto &row = mel_vector[i];
      Eigen::Map<Vectorf>(row.data(), row.size()) = mel.row(i);
    }
    return mel_vector;
  }

  /// \brief      compute mfcc similar with librosa.feature.mfcc
  /// \param      x             input audio signal
  /// \param      sr            sample rate of 'x'
  /// \param      n_fft         length of the FFT size
  /// \param      n_hop         number of samples between successive frames
  /// \param      win           window function. currently only supports 'hann'
  /// \param      center        same as librosa
  /// \param      mode          pad mode. support "reflect","symmetric","edge"
  /// \param      power         exponent for the magnitude melspectrogram
  /// \param      n_mels        number of mel bands
  /// \param      f_min         lowest frequency (in Hz)
  /// \param      f_max         highest frequency (in Hz)
  /// \param      n_mfcc        number of mfccs
  /// \param      norm          ortho-normal dct basis
  /// \param      type          dct type. currently only supports 'type-II'
  /// \return     mfcc matrix
  static std::vector<std::vector<float>> mfcc(std::vector<float> &x, int sr,
                                              int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode,
                                              float power, int n_mels, int fmin, int fmax,
                                              int n_mfcc, bool norm, int type) {
    Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
    Matrixf mel = librosa::mel::melspectrogram(map_x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax).transpose();
    Matrixf mel_db = librosa::util::power2db(mel);
    Matrixf dct = librosa::mel::dct(mel_db, norm, type).leftCols(n_mfcc);
    std::vector<std::vector<float>> mfcc_vector(dct.rows(), std::vector<float>(dct.cols(), 0.f));
    for (int i = 0; i < dct.rows(); ++i) {
      auto &row = mfcc_vector[i];
      Eigen::Map<Vectorf>(row.data(), row.size()) = dct.row(i);
    }
    return mfcc_vector;
  }
};

} // namespace librosa

#endif