#include <vector>
#include <caffex.h>
#include "adsb2.h"

namespace adsb2 {
    class CaffeDetector: public Detector {
        caffex::Caffex impl;
    public:
        CaffeDetector (string const &path)
            : impl(path) {
        }
        virtual void apply (cv::Mat image, cv::Mat *o) {
            CHECK(image.type() == CV_32F);
            cv::Mat u8;
            image.convertTo(u8, CV_8UC1);
            std::vector<float> prob;
            impl.apply(u8, &prob);
            //std::cerr << prob.size() << ' ' << input.total() << std::endl;
            BOOST_VERIFY(prob.size() == u8.total() * 2);
            cv::Mat m(u8.size(), CV_32F, &prob[u8.total()]);
            *o = m.clone();
        }
    };

    Detector *make_caffe_detector (string const &path) {
        return new CaffeDetector(path);
    }
}
