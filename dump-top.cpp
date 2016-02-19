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
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string list_path;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("list_path", po::value(&list_path), "")
    ;

    po::positional_options_description p;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);
    Cook cook(config);

    fs::path root("image");
    fs::ofstream meta("image.list");
    ifstream list(list_path);
    int n;
    while (list >> n) {
        fs::path ndir(lexical_cast<string>(n));
        fs::path study_path(fs::path("train")/ndir/fs::path("study"));
        fs::path out_dir(root/ndir);
        fs::create_directories(out_dir);
        Study study(study_path, true, true, true);
        cook.apply(&study);
        int cnt = 0;
        for (int i = study.size() - 3; i < study.size(); ++i) {
            auto const &ss = study[i];
            int mid = ss.size() / 2;
            cerr << mid << endl;
            cv::Mat gif1 = ss[0].image;
            cv::Mat gif2 = ss[mid].image;
            for (auto &slice: ss) {
                fs::path img_path(out_dir/fs::path(fmt::format("{}.gif", cnt++)));
                Series sr;
                sr.resize(2);
                cv::hconcat(slice.image, gif1, sr[0].image);
                cv::hconcat(slice.image, gif2, sr[1].image);
                type_convert(&sr[0].image, CV_8U);
                type_convert(&sr[1].image, CV_8U);
                sr.save_gif(img_path.native(), 40);
                meta << img_path.native() << '\t' << slice.path.native() << endl;
            }
        }
    }

    return 0;
}

