#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "adsb2.h"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;
using namespace adsb2;


namespace ba = boost::accumulators;
typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::count, ba::tag::variance, ba::tag::moment<2>>> Acc;

void EM_binarify (Mat &mat, Mat *out) {
    Mat pixels;
    mat.convertTo(pixels, CV_64F);
    pixels = pixels.reshape(1, pixels.total());
    BOOST_VERIFY(pixels.total() == mat.total());
    Mat logL, labels, probs;
    EM em(2);
    em.train(pixels, noArray(), labels, noArray());
    cerr << pixels.total() << ' ' << pixels.rows << ' ' << labels.total() << endl;
    BOOST_VERIFY(labels.total() == mat.total());
    labels.convertTo(*out, CV_8UC1, 255);
    *out = out->reshape(1, mat.rows);
}

template <typename I>
void bound (I begin, I end, int *b, float margin) {
    float cc = 0;
    I it = begin;
    while (it < end) {
        cc += *it;
        if (cc >= margin) break;
        ++it;
    }
    *b = it - begin;
}

void bound (vector<float> const &v, int *x, int *w, float margin) {
    int b, e;
    bound(v.begin(), v.end(), &b, margin);
    bound(v.rbegin(), v.rend(), &e, margin);
    e = v.size() - e;
    if (e < b) e = b;
    *x = b;
    *w = e - b;
}

void bound (Mat const &image, Rect *rect, float th) {
    vector<float> X(image.cols, 0);
    vector<float> Y(image.rows, 0);
    float total = 0;
    CHECK(image.type() == CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        float const *row = image.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x) {
            float v = row[x];
            X[x] += v;
            Y[y] += v;
            total += v;
        }
    }
    float margin = total * (1.0 - th) / 2;
    bound(X, &rect->x, &rect->width, margin);
    bound(Y, &rect->y, &rect->height, margin);
}

int main(int argc, char **argv) {
    //Stack stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;
    string gif;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    //("output,o", po::value(&output_dir), "")
    ("gif", po::value(&gif), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    //p.add("output", 1);
    //p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_dir.empty()) {
        cerr << desc;
        return 1;
    }

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    ImageLoader loader(config);
    DcmStack images(input_dir, loader);

    Detector *det = make_caffe_detector(config.get<string>("adsb2.caffe.model", "model"));
    CHECK(det) << " cannot create detector.";

    float th = config.get<float>("adsb2.bound_th", 0.95);

    for (auto &image: images) {
        Mat prob;
        det->apply(image, &prob);
        Rect bb;
        bound(prob, &bb, th);
        cv::rectangle(image, bb, cv::Scalar(0xFF));
    }

    delete det;

    if (gif.size()) {
        images.make_gif(gif);
    }
    return 0;

#if 0
    Size shape = images[0].size();
    unsigned pixels = images[0].total();

    Mat mu(shape, CV_32F);
    Mat sigma(shape, CV_32F);
    //Mat spread(shape, CV_32F);
    {
        vector<Acc> accs(pixels);
        for (auto const &image: images) {
            float const *v = image.ptr<float>(0);
            for (auto &acc: accs) {
                acc(*v);
                ++v;
            }
        }
        float *m = mu.ptr<float>(0);
        float *s = sigma.ptr<float>(0);
        //float *sp = spread.ptr<float>(0);
        for (auto const &acc: accs) {
            *m = ba::mean(acc);
            *s = std::sqrt(ba::variance(acc));
            //cout << *s << endl;
            //*sp = ba::max(acc) - ba::min(acc);
            ++m; ++s; //++sp;
        }
    }
    //sigma = images[0];
    Mat norm;
    normalize(sigma, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    imwrite("/home/wdong/public_html/a.jpg", norm);
    EM_binarify(sigma, &norm);
    //threshold(norm, norm, 64, 255, THRESH_BINARY);
    imwrite("/home/wdong/public_html/b.jpg", norm);

    return 0;
#endif
}

