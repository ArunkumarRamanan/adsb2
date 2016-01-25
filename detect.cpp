#include <queue>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "adsb2.h"

namespace fs = boost::filesystem;

using namespace std;
using namespace cv;
using namespace adsb2;

int conn_comp (cv::Mat *mat, cv::Mat const &weight, vector<float> *cnt) {
    // return # components
    CHECK(mat->type() == CV_8UC1);
    CHECK(mat->isContinuous());
    CHECK(weight.type() == CV_32F);
    cv::Mat out(mat->size(), CV_8UC1, cv::Scalar(0));
    CHECK(out.isContinuous());
    uint8_t const *ip = mat->ptr<uint8_t const>(0);
    uint8_t *op = out.ptr<uint8_t>(0);
    float const *wp = weight.ptr<float const>(0);

    int c = 0;
    int o = 0;
    if (cnt) cnt->clear();
    for (int y = 0; y < mat->rows; ++y)
    for (int x = 0; x < mat->cols; ++x) {
        do {
            if (op[o]) break;
            if (ip[o] == 0) break;
            // find a new component
            ++c;
            queue<int> todo;
            op[o] = c;
            todo.push(o);
            float W = 0;
            while (!todo.empty()) {
                int t = todo.front();
                todo.pop();
                W += wp[t];
                // find neighbors of t and add
                int tx = t % mat->cols;
                int ty = t / mat->cols;
                for (int ny = std::max(0, ty-1); ny <= std::min(mat->rows-1,ty+1); ++ny) 
                for (int nx = std::max(0, tx-1); nx <= std::min(mat->cols-1,tx+1); ++nx) {
                    // (ny, ix) is connected
                    int no = t + (ny-ty) * mat->cols + (nx-tx);
                    if (op[no]) continue;
                    if (ip[no] == 0) continue;
                    op[no] = c;
                    todo.push(no);
                }
            }
            if (cnt) cnt->push_back(W);
        } while (false);
        ++o;
    }
    *mat = out;
    CHECK(c == cnt->size());
    return c;
}

void post_process (Stack &stack, int mk) {
    cv::Mat p(stack.front().image.size(), CV_32F, cv::Scalar(0));
    for (auto &s: stack) {
        p = cv::max(p, s.prob);
    }
    cv::normalize(p, p, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::threshold(p, p, 100, 255, cv::THRESH_BINARY);
    vector<float> cc;
    conn_comp(&p, stack.front().vimage, &cc);
    CHECK(cc.size());
    float max_c = *std::max_element(cc.begin(), cc.end());
    for (unsigned i = 0; i < cc.size(); ++i) {
        if (cc[i] * 2 < max_c) cc[i] = 0;
    }
    for (int y = 0; y < p.rows; ++y) {
        uint8_t *ptr = p.ptr<uint8_t>(y);
        for (int x = 0; x < p.cols; ++x) {
            uint8_t c = ptr[x];
            if (c == 0) continue;
            --c;
            CHECK(c < cc.size());
            ptr[x] = cc[c] ? 1: 0;
        }
    }
    cv::Mat kernel = cv::Mat::ones(mk, mk, CV_8U);
    cv::dilate(p, p, kernel);
    cv::Mat np;
    p.convertTo(np, CV_32F);
    for (auto &s: stack) {
        cv::Mat prob = s.prob.mul(np);
        s.prob = prob;
    }
    // find connected components of p
}

int main(int argc, char **argv) {
    //Stack stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;
    string gif;
    float th;
    int mk;
    bool do_prob = false;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    //("output,o", po::value(&output_dir), "")
    ("gif", po::value(&gif), "")
    ("th", po::value(&th)->default_value(0.99), "")
    ("mk", po::value(&mk)->default_value(20),"")
    ("prob", "")
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
    Stack stack(input_dir);
    cook.apply(&stack);

    Detector *det = make_caffe_detector(config);
    CHECK(det) << " cannot create detector.";

    float bbth = config.get<float>("adsb2.bound_th", 0.95);
    for (auto &s: stack) {
        det->apply(&s);
    }
    delete det;
    post_process(stack, mk);

    for (auto &s: stack) {
        cout << s.path.native() << '\t' << sqrt(cv::sum(s.prob)[0]) * s.meta.spacing << endl;
        if (gif.size()) {
            Rect bb;
            bound(s.prob, &bb, bbth);
            cv::rectangle(s.image, bb, cv::Scalar(0xFF));
            if (do_prob) {
                cv::normalize(s.prob, s.prob, 0, 255, cv::NORM_MINMAX, CV_32FC1);
                cv::rectangle(s.prob, bb, cv::Scalar(0xFF));
                cv::hconcat(s.image, s.prob, s.image);
            }
        }
    }

    if (gif.size()) {
        stack.save_gif(gif);
    }
    return 0;
}

