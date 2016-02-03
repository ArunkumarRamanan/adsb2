#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <boost/multi_array.hpp>
#include <glog/logging.h>
#include "adsb2.h"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;
using namespace adsb2;

class DpSeg {
    struct E {
        float opt;    
        int prev;    // prev slice
    };
    typedef boost::multi_array<E, 2> WorkSpaceBase;
    class WorkSpace:public WorkSpaceBase {
    public:
        WorkSpace (int rows, int cols): WorkSpaceBase(boost::extents[rows][cols]) {
        }
    };
    int margin;
    float thr;
    float smooth;
    float wall;
    float penalty (int dx) const {
        return smooth *abs(dx);
    };
    Detector *caffe;
public:
    DpSeg (Config const &conf)
        : margin(conf.get<int>("adsb2.dp.margin", 5)),
        thr(conf.get<float>("adsb2.dp.th", 0.3)),
        smooth(conf.get<float>("adsb2.dp.smooth", 200)),
        wall(conf.get<float>("adsb2.dp.wall", 300)),
        caffe(nullptr)
    {
        int prob = conf.get<int>("adsb2.dp.prob", 0);
        if (prob) {
            caffe = make_caffe_detector(conf.get<string>("adsb2.dp.model", "contour-model"));
        }
    }
    ~DpSeg () {
        if (caffe) delete caffe;
    }
    void apply (cv::Mat input, vector<int> *seg, cv::Mat *vis) const {
        CHECK(input.type() == CV_32F);
        CHECK(input.cols >= margin *2);
        float th;
        cv::Mat image;
        if (caffe) {
            th = 255 * (1 - thr);
            cv::Mat u8;
            input.convertTo(u8, CV_8U);
            equalizeHist(u8, u8);
            int m = u8.rows / 4;
            cv::Mat top = u8(cv::Rect(0, 0, u8.cols, m));
            cv::Mat bot = u8(cv::Rect(0, u8.rows - m, u8.cols, m));
            cv::vconcat(bot, u8, u8);
            cv::vconcat(u8, top, u8);
            cv::Mat extended;
            caffe->apply(u8, &extended);
            image = extended(cv::Rect(0, m, input.cols, input.rows)).clone();
            image *= 255;
            *vis = image;
        }
        else {
            float left_mean = cv::mean(input(cv::Rect(0, 0, margin, input.rows)))[0];
            float right_mean = cv::mean(input(cv::Rect(input.cols - margin, 0, margin, input.rows)))[0];
            //cerr << left_mean << ' ' << right_mean << endl;
            CHECK(right_mean < left_mean);
            th = left_mean + (right_mean - left_mean) * thr;
            image = input;
            *vis = input;
        }

        WorkSpace ws(image.rows, image.cols);
        int best_cc = 0;
        for (int y = 0; y < image.rows; ++y) {
            CHECK(y < image.rows);
            E *e = &(ws[y][0]);
            float const *I = image.ptr<float const>(y);
            if (y == 0) {
                float acc = 0;
                for (int x = 0; x < image.cols; ++x) {
                    float delta = I[x] - th;
                    if (delta < 0) delta *= wall;
                    acc += delta;
                    e[x].opt = acc;
                    e[x].prev = -1;
                }
                continue;
            }
            E *prev = &(ws[y-1][0]);
            float acc = 0;
            best_cc = 0;
            float best_cc_score = -1;
            for (int x = 0; x < image.cols; ++x) {
                float delta = I[x] - th;
                if (delta < 0) delta *= wall;
                acc += delta;
                int lb = std::max(x - 7, 0);
                int ub = std::min(x + 7, image.cols-1);
                float best_score = -1;
                int best_prev = 0;
                for (int p = lb; p <= ub; ++p) {
                    float score = prev[p].opt + acc - penalty(p - x);
                    if (score > best_score) {
                        best_score = score;
                        best_prev = p;
                    }
                }
                e[x].opt = best_score;
                e[x].prev = best_prev;
                if ((x == 0) || (best_score > best_cc_score)) {
                    best_cc_score = best_score;
                    best_cc = x;
                }
            }
        }
        int y = image.rows - 1;
        while (y >= 0) {
            seg->push_back(best_cc);
            best_cc = ws[y][best_cc].prev;
            --y;
        }
        CHECK(best_cc = -1);
        std::reverse(seg->begin(), seg->end());
    }
};

void SCC_Analysis (Series *s, Config const &conf) {
    cv::Rect_<float> lb = unround(s->front().pred);
    cv::Rect_<float> ub = lb;
    for (auto &ss: *s) {
        cv::Rect_<float> r = unround(ss.pred);
        lb &= r;
        ub |= r;
    }
    cv::Point2f c(lb.x + 0.5 * lb.width, lb.y + 0.5 * lb.height);
    float R = max_R(c, ub) * 2;
#pragma parallel
    {
        DpSeg seg(conf);
#pragma omp for schedule(dynamic, 1)
        for (unsigned i = 0; i < s->size(); ++i) {
            Slice &ss = s->at(i);
            cv::Mat polar, cart;
            linearPolar(ss.image, &polar, c, R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
            vector<int> curve;
            cv::Mat vis;
            seg.apply(polar, &curve, &vis);
            for (int i = 1; i < curve.size(); ++i) {
                cv::line(vis, cv::Point(curve[i-1], i-1), cv::Point(curve[i], i), cv::Scalar(0xFF));
                cv::line(polar, cv::Point(curve[i-1], i-1), cv::Point(curve[i], i), cv::Scalar(0));
            }
            linearPolar(polar, &cart, c, R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);
            cv::Mat tmp;
            cv::hconcat(vis, polar, tmp);
            cv::hconcat(tmp, cart, ss.image);
            //ss.image = polar;
        }
    }
}

int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;
    string gif;
    bool do_prob = false;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    ("gif", po::value(&gif), "")
    ("prob", "")
    ;


    po::positional_options_description p;
    p.add("input", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_dir.empty()) {
        cerr << desc;
        return 1;
    }

    if (vm.count("prob")) do_prob = true;

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);

    Cook cook(config);
    Series stack(input_dir, true, true);
    cook.apply(&stack);

    float bbth = config.get<float>("adsb2.bound_th", 0.95);
#pragma omp parallel
    {
        Detector *det = make_caffe_detector(config);
        CHECK(det) << " cannot create detector.";
#pragma omp for schedule(dynamic, 1)
        for (unsigned i = 0; i < stack.size(); ++i) {
            det->apply(&stack[i]);
        }
        delete det;
    }
    MotionFilter(&stack, config);
#pragma omp parallel for
    for (unsigned i = 0; i < stack.size(); ++i) {
        auto &s = stack[i];
        FindSquare(s.prob, &s.pred, config);
    }
    SCC_Analysis(&stack, config);
    if (gif.size()) {
        //stack.visualize(do_prob);
        stack.save_gif(gif);
    }
    return 0;
}


