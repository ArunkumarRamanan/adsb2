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

namespace fs = boost::filesystem;

using namespace std;

class Stack: public std::vector<cv::Mat> {
public:
    cv::Size size () const {
        return at(0).size();
    }

    void convert (int rtype, double alpha = 1, double beta = 0) {
        for (auto &image: *this) {
            image.convertTo(image, rtype, alpha, beta);
        }
    }
};

class DcmStack: public Stack {
    fs::path input_dir;
    fs::path temp_dir;
    std::vector<fs::path> names;
public:
    DcmStack (std::string const &dir, std::string const &tmp)
        : input_dir(dir),
          temp_dir(tmp)
    {
        // enumerate DCM files
        fs::directory_iterator end_itr;
        fs::create_directories(temp_dir);
        for (fs::directory_iterator itr(input_dir);
                itr != end_itr; ++itr) {
            if (fs::is_regular_file(itr->status())) {
                // found subdirectory,
                // create tagger
                auto path = itr->path();
                auto stem = path.stem();
                auto ext = path.extension();
                if (ext.string() != ".dcm") {
                    LOG(WARNING) << "Unknown file type: " << path.string();
                    continue;
                }
                names.push_back(stem);
            }
        }
        std::sort(names.begin(), names.end());
        // convert to PGM and load
        ostringstream gif_cmd;
        gif_cmd << "convert -delay 5 ";
        resize(names.size());
        for (unsigned i = 0; i < names.size(); ++i) {
            auto const &name = names[i];
            auto dcm_path = input_dir;
            dcm_path /= name;
            dcm_path += ".dcm";
            auto pgm_path = temp_dir;
            pgm_path /= name;
            pgm_path += ".pgm";
            std::ostringstream cvt_cmd;
            cvt_cmd << "convert " << dcm_path << " " << pgm_path;
            ::system(cvt_cmd.str().c_str());
            gif_cmd << " " << pgm_path;

            cv::Mat image = cv::imread(pgm_path.string(), -1);
            BOOST_VERIFY(image.total());
            BOOST_VERIFY(image.type() == CV_16U);
            BOOST_VERIFY(image.isContinuous());
            if (i) {
                BOOST_VERIFY(image.size() == at(0).size());
            }
            at(i) = image;
        }
        fs::path gif_path = temp_dir;
#if 0
        gif_path /= input_dir.stem();
        gif_path += ".gif";
#endif
        gif_path /= "animation.gif";
        gif_cmd << " " << gif_path;
        ::system(gif_cmd.str().c_str());
    }
};

using namespace std;
using namespace cv;


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

int main(int argc, char **argv) {
    //Stack stack("sax", "tmp");
    namespace po = boost::program_options; 
    string input_dir;
    string output_dir;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("input,i", po::value(&input_dir), "")
    ("output,o", po::value(&output_dir), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);
    //p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_dir.empty() || output_dir.empty()) {
        cerr << desc;
        return 1;
    }

    DcmStack images(input_dir, output_dir);
    images.convert(CV_32F);

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
}

