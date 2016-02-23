#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "adsb2.h"

using namespace std;
using namespace adsb2;

void dump (vector<SliceReport> const &r, vector<SliceReport> const &r1, int label) {
    for (unsigned i = 0; i < r.size(); ++i) {
        auto &s = r[i];
        cout << label;
        cout << " 0:" << s.data[SL_BSCORE];
        cout << " 1:" << s.data[SL_PSCORE];
        cout << " 2:" << s.data[SL_CSCORE];
        cout << " 3:" << s.data[SL_CCOLOR];
        cout << " 4:" << s.data[SL_AREA] / r1[i].data[SL_AREA];
        cout << endl;
    }
}

int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    vector<string> paths;
    float scale;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&paths), "")
    ("scale,s", po::value(&scale)->default_value(20), "")
    ("detail", "")
    ;


    po::positional_options_description p;
    p.add("input", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || paths.empty()) {
        cerr << desc;
        return 1;
    }
    bool detail = vm.count("detail") > 0;
    Eval eval;
    vector<float> v;
    for (auto const &s: paths) {
        fs::path p(s);
        StudyReport x(p);
        unsigned sz = x.size();
        unsigned pos = sz - 1;
        unsigned neg = sz - 3;
        dump(x[pos], x[pos-1], 1);
        dump(x[neg], x[neg-1], 0);
    }
}

