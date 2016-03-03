#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/covariance.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "adsb2.h"

namespace adsb2 {
    extern fs::path home_dir;
}

using namespace std;
using namespace adsb2;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    vector<int> studies;
    fs::path data_root; // report directory

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&studies), "report files")
    ("data", po::value(&data_root), "dir containing report files")
    ;

    po::positional_options_description p;
    p.add("data", 1);
    p.add("input", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || data_root.empty()) {
        cerr << "ADSB2 VERSION: " << VERSION << endl;
        cerr << desc;
        return 1;
    }

    if (studies.empty()) {
        int study;
        while (cin >> study) {
            studies.push_back(study);
        }
    }

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);
    GlobalInit(argv[0], config);

    for (auto const &study: studies) {
        fs::path path(data_root / fs::path(lexical_cast<string>(study))/ fs::path("report.txt"));
        StudyReport x(path);
        if (x.empty()) {
            continue;
        }
        for (auto const &sax: x) {
            vector<float> v;
            for (auto const &s: sax) {
                v.push_back(s.data[SL_AREA]);
            }
            if (v.size() < 4) continue;
            sort(v.begin(), v.end());
            cout << v[0] << '\t' << v[1] << '\t' << v[v.size()-3] << '\t' << v[v.size()-1] << endl;

        }
    }

    return 0;
}

