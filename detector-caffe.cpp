#include <vector>
#include <caffex.h>
#include "adsb2.h"

namespace adsb2 {
    class CaffeDetector: public Detector {
        caffex::Caffex impl;
    public:
        CaffeDetector (string const &path)
            : impl(path, caffe_batch) {
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
        virtual void apply (vector<cv::Mat> &images, vector<cv::Mat> *o) {
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
            cv::Mat all;
            impl.apply(images, &all);
            o->resize(images.size());
            int total = images[0].total();
            CHECK(all.rows == o->size());
            CHECK(all.cols == total * 2);
            for (int i = 0; i < all.rows; ++i) {
                float *ptr = all.ptr<float>(i);
                cv::Mat m(images[0].size(), CV_32F, ptr + total);
                o->at(i) = m.clone();
            }
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
