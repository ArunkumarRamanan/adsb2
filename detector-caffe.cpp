#include <vector>
#include <caffex.h>
#include "adsb2.h"

namespace adsb2 {
    class CaffeDetector: public Detector {
        caffex::Caffex impl;
    public:
        CaffeDetector (string const &model)
            : impl(model) {
        }
        virtual void apply (cv::Mat input, cv::Mat *output) {
            std::vector<float> prob;
            impl.apply(input, &prob);
            //std::cerr << prob.size() << ' ' << input.total() << std::endl;
            BOOST_VERIFY(prob.size() == input.total() * 2);
            cv::Mat m(input.size(), CV_32F, &prob[input.total()]);
            *output = m.clone();
        }
    };

    Detector *make_caffe_detector (string const &dir) {
        return new CaffeDetector(dir);
    }
}
