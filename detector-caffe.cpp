#include <vector>
#include <caffex.h>
#include "adsb2.h"

namespace adsb2 {
    class CaffeDetector: public Detector {
        caffex::Caffex impl;
        int channels;
    public:
        CaffeDetector (Config const &config) //string const &model)
            : impl(config.get<string>("adsb2.caffe.model", "model")),
            channels(config.get<int>("adsb2.caffe.channels", 1))
        {
        }
        virtual void apply (Slice *sample) {
            cv::Mat u8;
            CaffeAdaptor::apply(*sample, &u8, nullptr, channels);
            std::vector<float> prob;
            impl.apply(u8, &prob);
            //std::cerr << prob.size() << ' ' << input.total() << std::endl;
            BOOST_VERIFY(prob.size() == u8.total() * 2);
            cv::Mat m(u8.size(), CV_32F, &prob[u8.total()]);
            sample->prob = m.clone();
        }
    };

    Detector *make_caffe_detector (Config const &config) {
        return new CaffeDetector(config);
    }
}
