#ifndef LIBROSA_H_
#define LIBROSA_H_
// #define EIGEN_USE_MKL_ALL
// #define EIGEN_VECTORIZE_SSE4_2

#include <sndfile.h>
#include <samplerate.h>
#include <string.h>

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>

#include <vector>
#include <unordered_map>
#include <complex>
#include <iostream>
#include <memory>
#include <cmath>

///
/// \brief c++ implemention of librosa
///
namespace librosa{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // !M_PI

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor> Vectord;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcf;

namespace internal{

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

  static const double BW_FASTEST = 0.85;
  static const double BW_BEST = 0.94759372;

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

  static Matrixcf stft(const Vectorf &x, int n_fft, int n_hop, const std::string &win, bool center, const std::string &mode){
    // hanning
    Vectorf window = get_window(win, n_fft);

    int pad_len = center ? n_fft / 2 : 0;
    Vectorf x_paded = pad(x, pad_len, pad_len, mode, 0.f);

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

  static Matrixf spectrogram(Matrixcf &X, float power = 1.f){
    // cwiseAbs求每个元素的模 pow 幂
    return X.cwiseAbs().array().pow(power);
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

  static Matrixf melspectrogram(Vectorf &x, int sr, int n_fft, int n_hop,
                          const std::string &win, bool center,
                          const std::string &mode, float power,
                          int n_mels, int fmin, int fmax, bool htk = false){
    Matrixcf X = stft(x, n_fft, n_hop, win, center, mode);
    Matrixf mel_basis = melfilter(sr, n_fft, n_mels, fmin, fmax, htk);
    Matrixf sp = spectrogram(X, power);
    Matrixf mel = mel_basis*sp.transpose();
    return mel;
  }

  static Matrixf power2db(Matrixf& x) {
    auto log_sp = 10.0f*x.array().max(1e-10).log10();
    return log_sp.cwiseMax(log_sp.maxCoeff() - 80.0f);
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
      ortho[0] = sqrtf(0.25f/N);
      dct = dct*ortho.asDiagonal();
    }
    return dct;
  }

  static Vectorf cqt_frequencies(int n_bins, float f_min, int bins_per_octave=12, float tuning=0.0){
    float correction = std::pow(2.f, tuning/bins_per_octave);
    Vectorf frequencies = pow(2.f, Vectorf::LinSpaced(n_bins, 0, n_bins-1).array());
    return frequencies;
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
    int downsample_count1 = std::max(0, int(std::ceil(std::log2f(BW_FASTEST*nyquist/filter_cutoff)))-2);
    int num_twos = _num_two_factors(hop_length);
    int downsample_count2 = std::max(0, num_twos-n_octaves+1);
    return std::min(downsample_count1, downsample_count2);
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
    return new_y;
  }

  static Vectorf resample(const Vectorf& y, float orig_sr, float target_sr, 
                  const std::string& res_type="kaiser_best", bool fix=true, 
                  bool scale=false){
    
    if(std::abs(orig_sr-target_sr)<1e-6) return y;
    float ratio = target_sr / orig_sr;
    int n_samples = int(std::ceil(y.size()*ratio));
    if(n_samples<1){
      throw std::length_error("input signal is too small");
    }

    int filter_type;
    if(res_type=="kaiser_best"){
      filter_type = SRC_SINC_BEST_QUALITY;
    }
    else if(res_type=="kaiser_fast"){
      filter_type = SRC_SINC_FASTEST;
    }
    else{
      throw std::invalid_argument("wrong res_type");
    }

    auto data_in = std::unique_ptr<float>(new float[y.size()]);
    auto data_out = std::unique_ptr<float>(new float[n_samples]);
    std::copy(y.data(), y.data()+y.size(), data_in.get());

    SRC_DATA res_data;
    res_data.src_ratio = ratio;
    res_data.input_frames = y.size();
    res_data.output_frames = n_samples;
    res_data.data_in = data_in.get();
    res_data.data_out = data_out.get();

    // 最后一个参数int channels
    if(src_simple(&res_data, filter_type, 1))
    {
      std::cout << "src_simple failed." << std::endl;
      exit(1);
    }

    Vectorf target_y = Eigen::Map<Vectorf>(data_out.get(), n_samples);
    if(fix){
      target_y = fix_length(target_y, n_samples);
    }
    if(scale){
      target_y /= std::sqrt(ratio);
    }
    return target_y;
  }

  static void __early_downsample(Vectorf& y, float&sr, int& hop_length, std::string res_type,
                          int n_octaves, float nyquist, float filter_cutoff, bool scale){
    int downsample_count = __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves);
    if(downsample_count>0 && res_type=="kaiser_fast"){
      int downsample_factor = int(std::pow(2, downsample_count));
      hop_length /= downsample_factor;
      if(y.size()<downsample_factor){
        throw std::invalid_argument("Input signal y is too short");
      }
      float new_sr = sr / float(downsample_factor);
      y = resample(y, sr, new_sr, res_type, true, true);
      if(!scale) y*=std::sqrt(downsample_factor);
      sr = new_sr;
    }
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

    if(freq(n_bins-1)*(1+0.5*WINDOW_BANDWIDTHS[window]/Q) > sr/2.f){
      throw std::invalid_argument("Filter pass-band lies beyond Nyquist");
    }
    Vectorf lengths = Q*sr/freq.array();
    return lengths;
  }

  static void filters_constant_q(
                            Matrixcf& filters, Vectorf& lengths,
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
    filters.resize(lengths.size(), max_len);

    std::complex<float> a(0, 1);

    for(int i=0; i<lengths.size(); ++i){
      int ilen = int(lengths[i]);
      float freq = freqs[i];
      Vectorcf sig = exp(Vectorf::LinSpaced(ilen/2*2, static_cast<float>(-ilen/2), static_cast<float>(ilen/2))
                     .array()*a*2*M_PI*freq/sr);
      // 计算窗函数
      sig = sig*get_window(window, sig.size());
      // sig = normalize(sig, norm=norm); 完成normalize的步骤 按照行相加归一化
      float norm = sig.lpNorm<1>();
      if(norm > 1e-7){
        sig /= norm;
      }
      int left = (max_len - ilen)/2, right = max_len - ilen - left;
      filters.row(0) = pad(sig, left, right, "constant");
    }
  }

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
    filters_constant_q(basis, lengths, sr, fmin, n_bins, bins_per_octave, 
                      tuning, window, filter_scale, true, norm);
    int n_frames = basis.rows(), n_fft = basis.cols(), n_f = n_fft/2+1;
    if(hop_length>0 && n_fft < std::pow(2.f, 1+std::ceil(std::log2(hop_length)))){
      n_fft = int(std::pow(2.f, 1+std::ceil(std::log2f(hop_length))));
    }
    basis *= lengths / float(n_fft);

    Eigen::FFT<float> fft;
    for (int i = 0; i < n_frames; ++i){
      basis.row(i) = fft.fwd(basis.row(i));
    }
    fft_basis = basis.leftCols(n_f);
    return n_fft;
  }

  static Matrixcf __cqt_response(const Vectorf& y, 
                                int n_fft,
                                int hop_length,
                                const Matrixcf& fft_basis,
                                const std::string& mode){

    Matrixcf D = stft(y, n_fft, hop_length, "ones", true, mode);
    return fft_basis*D;
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
    int len_orig = y.size();

    if(fmin<0) fmin = 32.703196; // C1 fmin = note_to_hz("C1");
    // if(tuning<0) estimate_tuning(y, sr);

    Vectorf freqs = cqt_frequencies(n_bins, fmin, bins_per_octave).rightCols(bins_per_octave);
    float fmin_t = freqs.minCoeff();
    float fmax_t = freqs.maxCoeff();

    float Q = filter_scale / (std::pow(2.f, 1.f/bins_per_octave)-1);
    float filter_cutoff = fmax_t * (1+0.5*(WINDOW_BANDWIDTHS[window])/Q);
    float nyquist = sr/2.f;
    std::string res_type;
    if(filter_cutoff < BW_FASTEST*nyquist){
      res_type = "kaiser_fast";
    }else{
      res_type = "kaiser_best";
    }

    __early_downsample(y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale);
    
    Matrixcf cqt_resp(bins_per_octave*n_octaves, len_orig/hop_length+1);

    Matrixcf fft_basis;
    int n_fft;

    if(res_type!="kaiser_fast"){
      n_fft = __cqt_filter_fft(fft_basis, sr, fmin_t, n_filters, bins_per_octave, tuning,
                                  filter_scale, norm, window);
      cqt_resp.topRows(n_filters) = __cqt_response(y, n_fft, hop_length, fft_basis, pad_mode);

      fmin_t /= 2;
      fmax_t /= 2;
      n_octaves -= 1;
      
      filter_cutoff = fmax_t * (1+0.5*WINDOW_BANDWIDTHS[window]/Q);
      res_type = "kaiser_fast";
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
        my_y = resample(my_y, my_sr, my_sr / 2.f, res_type, true, true);
        fft_basis *= std::sqrt(2);
        my_sr /= 2.f;
        my_hop /= 2;
      }
      cqt_resp.topRows(n_filters) = __cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode);
    }

    Matrixcf C = cqt_resp.topRows(n_bins);

    if(scale){
      Vectorf lengths = constant_q_lengths(sr, fmin, n_bins, bins_per_octave, tuning, window, filter_scale);
      C = C.array() / lengths.array().sqrt();
    }
    return C;
  }

} // namespace internal

/// \brief      Load an audio file as a floating point time series
/// \param      path          path to the input file
/// \param      sr            target sampling rate
/// \param      mono          mono or dual
/// \param      offset        the begin pos of the audio
/// \param      duration      the length of the part want to get
/// \param      dtype         the data format of the return audio
/// \param      res_type      the filter type for resampling
 
static std::vector<float> load(std::string path,
                               int sr=22050,
                               bool mono=true,
                               float offset=0.f,
                               float duration=-1.f,
                               std::string dtype="float",
                               std::string res_type="kaiser_best"){
  SNDFILE *infile = NULL;
  SF_INFO in_sfinfo;
  memset(&in_sfinfo, 0, sizeof(in_sfinfo));
  if((infile=sf_open(path.c_str(), SFM_READ, &in_sfinfo))==NULL){
    std::cout << "Not able to open output file " << path << std::endl;
    sf_close(infile);
    exit(1);
  }
  if(in_sfinfo.format != (SF_FORMAT_WAV | SF_FORMAT_PCM_16)){ // 前者是主格式 后者是编码类型 
    std::cout << "the input file is not wav format!" << std::endl;
    exit(1);
  }

  float* data = new float[in_sfinfo.frames * in_sfinfo.channels];
  sf_readf_float(infile, data, in_sfinfo.frames);
  sf_close(infile);

  if(in_sfinfo.channels>1){
    for(int i=0;i<in_sfinfo.frames;++i){
      data[i] = std::accumulate(data+i*in_sfinfo.channels, data+(i+1)*in_sfinfo.channels, 0.0) / float(in_sfinfo.channels);
    }
  }

  float ratio = float(sr) / float(in_sfinfo.samplerate);
  int output_frames = std::ceil(in_sfinfo.frames*ratio);
  std::vector<float>audio(output_frames);

  if(std::abs(ratio-1)<1e-6){
    std::copy(data, data+in_sfinfo.frames, audio.begin());
  }
  else{
    SRC_DATA res_data;
    res_data.src_ratio = ratio;
    res_data.input_frames = in_sfinfo.frames;
    res_data.output_frames = output_frames;
    res_data.data_in = data ;
    res_data.data_out = audio.data();

    // 最后一个参数int channels
    if(src_simple(&res_data, SRC_SINC_BEST_QUALITY, 1))
    {
      std::cout << "src_simple failed." << std::endl;
      exit(1);
    }
  }
  delete []data;
  return audio;
}

// static Matrixf abs(Matrixcf& matrix){
//   return matrix.array().abs();
// }

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
    Matrixcf X = internal::stft(map_x, n_fft, n_hop, win, center, mode);
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
    Matrixcf X = internal::cqt(map_x, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm,
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
    Matrixf X = internal::cqt(map_x, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm,
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
    Matrixf mel = internal::melspectrogram(map_x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax, htk).transpose();
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
    Matrixf mel = internal::melspectrogram(map_x, sr, n_fft, n_hop, win, center, mode, power, n_mels, fmin, fmax).transpose();
    Matrixf mel_db = internal::power2db(mel);
    Matrixf dct = internal::dct(mel_db, norm, type).leftCols(n_mfcc);
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