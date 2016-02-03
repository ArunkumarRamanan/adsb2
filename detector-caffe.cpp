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
        CaffeDetector (string const &path)
            : impl(path), channels(1) {
        }
        virtual void apply (cv::Mat image, cv::Mat *o) {
            cv::Mat u8;
            if (image.type() == CV_8UC1) {
                u8 = image;
            }
            else {
                image.convertTo(u8, CV_8UC1);
            }
            std::vector<float> prob;
            impl.apply(u8, &prob);
            //std::cerr << prob.size() << ' ' << input.total() << std::endl;
            BOOST_VERIFY(prob.size() == u8.total() * 2);
            cv::Mat m(u8.size(), CV_32F, &prob[u8.total()]);
            *o = m.clone();
        }
            
            //string const &model)
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

    Detector *make_caffe_detector (string const &path) {
        return new CaffeDetector(path);
    }
}
