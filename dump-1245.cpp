#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "adsb2.h"

using namespace std;
using namespace boost;
using namespace cv;
using namespace adsb2;

int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string list_path;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("list", po::value(&list_path), "")
    ;

    po::positional_options_description p;
    p.add("list", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || (vm.count("list") == 0)) {
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
    Cook cook(config);

    fs::path root1("image0");
    fs::path root0("image1");
    ifstream list(list_path);
    int n;
    while (list >> n) {
        fs::path ndir(lexical_cast<string>(n));
        fs::path study_path(fs::path("train")/ndir/fs::path("study"));
        Study study(study_path, true, true, true);
        cook.apply(&study);
        int cnt = 0;
        for (int i = 0; i < 5; ++i) {   // 0,1  -- 3,4
            if (i >= study.size()) break;
            if (i == 2) continue;
            fs::path root = i < 2 ? root1 : root0;
            fs::path out_dir(root/ndir);
            fs::create_directories(out_dir);
            auto const &ss = study[i];
            int mid = ss.size() / 2;
            cerr << mid << endl;
            for (auto &slice: ss) {
                fs::path img_path(out_dir/fs::path(fmt::format("{}.jpg", cnt++)));
                imwrite(img_path.native(), slice.images[IM_IMAGE]);
            }
        }
    }

    return 0;
}

