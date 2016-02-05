#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include <dai/alldai.h>
#include "adsb2.h"

using namespace std;
using namespace boost;
using namespace cv;
using namespace adsb2;

int var_id (int slice, int row, int nvar_per_slice);

class VarEnc {
    int n; // = series->front().polar_prob.rows;
    int total;
public:
    VarEnc (Series const &series) {
        n = series.front().polar_prob.rows;
        total = n * series.size();
    }
    int operator () (int slice, int row) {
        return n * slice + row;
    }
    int size () const {
        return total;
    }
};

void fill_cost (cv::Mat *c, float r) {
    CHECK(c->type() == CV_64F);
    for (int y = 0; y < c->rows; ++y) {
        double *row = c->ptr<double>(y);
        for (int x = 0; x < c->cols; ++x) {
            row[x] = c->rows * r -r * std::abs(x - y);
        }
    }
}

void ca2 (Series *series, int max_it, float th = 0.3 * 255) {
    float wall = 300;
    int nstate = series->front().polar_prob.cols;
    VarEnc venc(*series);

    cv::Mat scost (nstate, nstate, CV_64F);
    cv::Mat tcost (nstate, nstate, CV_64F);

    fill_cost(&scost, 200);
    fill_cost(&tcost, 400);

    vector<dai::Var> vars(venc.size());
    vector<dai::Factor> factors;

    for (int sid = 0; sid < series->size(); ++sid) {
        Slice &s = series->at(sid);
        for (int y = 0; y < s.polar_prob.rows; ++y) {
            int vid = venc(sid, y);
            vars[vid] = dai::Var(venc(sid, y), nstate);
            float const *row = s.polar_prob.ptr<float const>(y);
            vector<double> cost(nstate);
            double acc = 0;
            for (int x = 0; x < s.polar_prob.cols; ++x) {
                double delta = row[x] - th;
                if (delta < 0) delta *= wall;
                acc += delta;
                cost[x] = acc;
            }
            factors.emplace_back(vars[vid], cost);
        }
    }
    for (int sid = 0; sid < series->size(); ++sid) {
        Slice &s = series->at(sid);
        for (int y = 0; y < s.polar_prob.rows; ++y) {
            int vid = venc(sid, y);
            if (y > 0) {
                factors.emplace_back(dai::VarSet(vars[vid], vars[venc(sid, y-1)]), scost.ptr<double>(0));
            }
            if (sid > 0) {
                factors.emplace_back(dai::VarSet(vars[vid], vars[venc(sid-1, y)]), tcost.ptr<double>(0));
            }
        }
    }
    cerr << vars.size() << " variables." << endl;
    cerr << factors.size() << " factors." << endl;

    dai::FactorGraph fg (factors.begin(), factors.end(), vars.begin(), vars.end(), factors.size(), vars.size());

    string algo = "BP[updates=SEQMAX,maxiter=1,tol=1e-9,logdomain=0]";
    dai::InfAlg *ia = dai::newInfAlgFromString(algo, fg);
    ia->init();
    vector<double> m(fg.nrVars(), 0.0);
    for (int it = 0; it < max_it; ++it) {
        ia->setMaxIter(it + 1);
        ia->run();
        cerr << '.';
    }
    std::vector<size_t> sol = ia->findMaximum();
    CHECK(sol.size() == venc.size());
    {
        int off = 0;
        for (int sid = 0; sid < series->size(); ++sid) {
            Slice &s = series->at(sid);
            s.polar_contour.resize(nstate);
            for (auto &v: s.polar_contour) {
                v = sol[off++];
            }
        }
    }
    cerr << endl;
    delete ia;
}


int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string gif_path;
    int sz, n;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    ("gif", po::value(&gif_path), "")
    (",s", po::value(&sz)->default_value(1), "")
    (",n", po::value(&n)->default_value(10), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("gif", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || input_dir.empty() || gif_path.empty()) {
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

    GlobalInit(argv[0], config);

    Series series;
    {
        fs::path dir(input_dir);
        fs::ifstream meta(dir / fs::path("meta"));
        CHECK(meta);
        Slice slice;
        int cnt = 0;
        string slice_path;
        while (meta >> slice_path >> slice.polar_R >> slice.polar_C.x >> slice.polar_C.y) {
            series.emplace_back();
            std::swap(series.back(), slice);
            series.back().path = fs::path(slice_path);
            fs::path vpath = dir / fs::path(fmt::format("image-{}.pgm", cnt));
            fs::path ppath = dir / fs::path(fmt::format("prob-{}.pgm", cnt));
            series.back().polar = cv::imread(vpath.native(), -1);
            series.back().polar_prob = cv::imread(ppath.native(), -1);
            type_convert(&series.back().polar, CV_32F);
            type_convert(&series.back().polar_prob, CV_32F);
            ++cnt;
        }
    }
    cerr << series.size() << " slices loaded." << endl;

    series.resize(sz);
    if (n) {
        ca2(&series, n);
    }

    for (auto &s: series) {
        cv::Mat cart;
        auto const &cc = s.polar_contour;
        for (int i = 1; i < cc.size(); ++i) {
            cv::line(s.polar, cv::Point(cc[i-1], i-1), cv::Point(cc[i], i), cv::Scalar(0), 2);
            cv::line(s.polar_prob, cv::Point(cc[i-1], i-1), cv::Point(cc[i], i), cv::Scalar(0), 2);
        }
        linearPolar(s.polar, &cart, s.polar_C, s.polar_R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
        hconcat3(cart, s.polar, s.polar_prob, &s.image);
        type_convert(&s.image, CV_8U);
    }

    series.save_gif(fs::path(gif_path));


    return 0;
}

