#ifndef AUDIO_H_
#define AUDIO_H_

#include <sndfile.h>
#include <samplerate.h>
#include <string.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <exception>

#include "util.h"

namespace librosa{
  namespace audio{

    static const double BW_FASTEST = 0.85;
    static const double BW_BEST = 0.94759372;

    static Vectorf resample(const Vectorf& y, float orig_sr, float target_sr, 
                const std::string& res_type="kaiser_best", bool fix=true, 
                bool scale=false){
  
      if(std::abs(orig_sr-target_sr)<1e-6) return y;
      float ratio = target_sr / orig_sr;
      std::cout << "ratio: " << ratio << std::endl;
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
        
        target_y = util::fix_length(target_y, n_samples);
      }

      if(scale){
        target_y /= std::sqrt(ratio);
      }
      return target_y;
    }

    static std::vector<float> load(const std::string& path,
                              int sr=22050,
                              bool mono=true,
                              float offset=0.f,
                              float duration=-1.f,
                              const std::string& dtype="float",
                              const std::string& res_type="kaiser_best"){
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

  }
}

#endif