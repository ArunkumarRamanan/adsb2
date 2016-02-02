#define CPU_ONLY 1
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <boost/scoped_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <json11.hpp>

#include "adsb2.h"

using namespace std;
using namespace boost;
using namespace cv;
using namespace json11;
using namespace adsb2;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string list_path;
    string root_dir;
    string output_dir;
    bool full = false;
    int F;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("list", po::value(&list_path), "")
    ("root", po::value(&root_dir), "")
    ("output,o", po::value(&output_dir), "")
    ;

    po::positional_options_description p;
    p.add("list", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || list_path.empty() || output_dir.empty()) {
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

    ifstream is(list_path);
    string line;
    fs::path out_dir(output_dir);
    fs::path out_list(out_dir);
    out_list += fs::path(".list");
    fs::create_directories(out_dir);
    unsigned cc = 0;
    fs::ofstream os(out_list);
    while (getline(is, line)) {
        Slice slice(line);
        fs::path orig = slice.path;
        slice.path = fs::path(root_dir)/orig;
        if (!fs::is_regular_file(slice.path)) {
            LOG(ERROR) << "cannot find file " << slice.path;
            continue;
        }
        slice.load_raw();
        slice.path = orig;
        CHECK(slice.annotated);
        // crop interested region
        cv::Point2f c(slice.box.x + 0.5 * slice.box.width,
                      slice.box.y + 0.5 * slice.box.height);
        float R = max_R(c, slice.box) * 2;
        cv::Mat image;
        slice.raw.convertTo(image, CV_32F);
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_32F);
        cv::Mat polar;
        linearPolar(image, &polar, c, R);
        cv::equalizeHist(polar, polar);
        cv::Mat out;
        polar.convertTo(out, CV_8UC1);
        fs::path outf(out_dir/fs::path(fmt::format("{}.png", cc++)));
        cv::imwrite(outf.native(), out);
        Json json = Json::object{
            {"dcm", slice.path.native()},
            {"x", c.x},
            {"y", c.y},
            {"R", R}
        };
        os << outf.native() << '\t' << json.dump() << endl;
    }

    return 0;
}

