#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <sndfile.h>
#include <string.h>
#include <samplerate.h>
#include "librosa.h"

int main(int argc, char* argv[]){
    if(argc!=2){
        std::cout << "Usage error! ./mel filepath " <<std::endl;
        return 1;
    }

    int n_fft = 2048;
    int n_hop = 512;
    int n_mel = 229;
    int fmin = 30;
    int fmax = 8000;
    int sr = 32000;

    auto x = librosa::load(argv[1], sr, true);
    auto melspectrogram_start_time =  std::chrono::system_clock::now();
    // 使用htk的频点分布公式
    std::vector<std::vector<float>> mels = librosa::Feature::melspectrogram(x, sr, n_fft, n_hop, "hann", true, "reflect", 2.f, n_mel, fmin, fmax, true);
    auto melspectrogram_end_time =  std::chrono::system_clock::now();
    auto melspectrogram_duration = std::chrono::duration_cast<std::chrono::milliseconds>(melspectrogram_end_time - melspectrogram_start_time);
    std::cout<<"Melspectrogram runing time is "<< melspectrogram_duration.count() << "ms" <<std::endl;

    // 查看mel谱相关的信息
    std::cout << "the mel height: " << mels.size() << std::endl
              << "the mel width: " << mels[0].size() << std::endl;

    auto accusum = [](float& a, float& b) { return a+b;};

    for(int i = 0 ; i < 10; i ++) {
        float sum = std::accumulate(mels[i].begin(), mels[i].end(), 0.f, accusum);
        std::cout << sum << " ";
    }
    std::cout << std::endl;

    // 计算cqt频谱
    auto cqtspec = librosa::Feature::cqtspectrogram(x, sr, n_hop, fmin, 256, 48);

    std::cout << "the cqt height: " << cqtspec.size() << std::endl
              << "the cqt width: " << cqtspec[0].size() << std::endl;
    
    for(int i=0; i<10; ++i) {
        float sum = std::accumulate(cqtspec[i].begin(), cqtspec[i].end(), 0.f, accusum);
        std::cout << sum << " ";
    }
    std::cout << std::endl;
    return 0;
}