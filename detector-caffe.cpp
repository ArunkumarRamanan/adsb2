#include <vector>
#include "caffex.h"
#include "heart.h"

namespace heart {
    class CaffeDetector: public Detector {
        caffex::Caffex impl;
        cv::Size size;  // caffe working image size
    public:
        CaffeDetector (string const &model)
            : impl(model) {
            size = impl.inputSize();
        }
        virtual void apply (cv::Mat input, cv::Mat *output) {
            std::vector<float> prob;
            impl.apply(input, &prob);
            BOOST_VERIFY(prob.size() == size.width * size.height);
            cv::Mat m(size, CV_32F, &prob[0]);
            cv::resize(m, *output, input.size());
        }
    };

    Detector *make_caffe_detector (string const &dir) {
        return new CaffeDetector(dir);
    }
}
