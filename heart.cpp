#include <iostream>
#include <opencv2/opencv.hpp>
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

using namespace std;
using namespace cv;

namespace ba = boost::accumulators;
typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::count, ba::tag::variance, ba::tag::moment<2>>> Acc;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    vector<string> paths;
    string output_dir;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("input", po::value(&paths), "")
    ("output,o", po::value(&output_dir), "")
    ;


    po::positional_options_description p;
    p.add("input", -1);
    //p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    cerr << paths.size() << endl;

    if (vm.count("help") || (paths.size() <= 1)) {
        cerr << desc;
        return 1;
    }

    vector<Mat> images(paths.size());

    for (unsigned i = 0; i< paths.size(); ++i) {
        images[i] = imread(paths[i], 0);
        BOOST_VERIFY(images[i].total());
        BOOST_VERIFY(images[i].type() == CV_8U);
        BOOST_VERIFY(images[i].isContinuous());
        if (i) {
            BOOST_VERIFY(images[i].size() == images[0].size());
        }
    }
    Size shape = images[0].size();
    unsigned pixels = images[0].total();

    Mat mu(shape, CV_32F);
    Mat sigma(shape, CV_32F);
    //Mat spread(shape, CV_32F);
    {
        vector<Acc> accs(pixels);
        for (auto const &image: images) {
            uint8_t const *v = image.ptr<uint8_t>(0);
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
    Mat norm;
    normalize(sigma, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    imwrite("a.jpg", norm);

    return 0;
}

