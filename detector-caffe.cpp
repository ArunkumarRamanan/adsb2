#include <vector>
#include <caffex.h>
#include "adsb2.h"

namespace adsb2 {
    class CaffeDetector: public Detector {
        caffex::Caffex impl;
    public:
        CaffeDetector (string const &path)
            : impl(path, 1) {
        }
        virtual void apply (cv::Mat image, cv::Mat *o) {
            /*
            cv::Mat u8;
            if (image.type() == CV_8U) {
                u8 = image;
            }
            else if (image.type() == CV_32F) {
                image.convertTo(u8, CV_8UC1);
            }
            else CHECK(0);
            */
            std::vector<float> prob;
            impl.apply(image, &prob);
            //std::cerr << prob.size() << ' ' << input.total() << std::endl;
            BOOST_VERIFY(prob.size() == image.total() * 2);
            cv::Mat m(image.size(), CV_32F, &prob[image.total()]);
            *o = m.clone();
        }
        virtual void apply (cv::Mat image, vector<float> *prob) {
            /*
            cv::Mat u8;
            if (image.type() == CV_8U) {
                u8 = image;
            }
            else if (image.type() == CV_32F) {
                image.convertTo(u8, CV_8UC1);
            }
            else CHECK(0);
            */
            impl.apply(image, prob);
        }
    };

    Detector *make_caffe_detector (fs::path const &path) {
        return new CaffeDetector(path.native());
    }
}
