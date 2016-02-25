#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/program_options.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "adsb2.h"

using namespace std;
using namespace boost;
using namespace cv;
using namespace adsb2;

class OsirisLocation {
public:
    static float compute (Slice const &s) {
        float pixelSpacingX = s.meta.raw_spacing;
        float pixelSpacingY = s.meta.raw_spacing;
        cv::Point3f ori_z = s.meta.ori_row.cross(s.meta.ori_col);
        float orientation[] = {
            s.meta.ori_row.x, s.meta.ori_row.y, s.meta.ori_row.z,
            s.meta.ori_col.x, s.meta.ori_col.y, s.meta.ori_col.z,
            ori_z.x, ori_z.y, ori_z.z,
        };
        float originX = s.meta.pos.x;
        float originY = s.meta.pos.y;
        float originZ = s.meta.pos.z;
        float x = s.meta.width / 2 - 0.5;
        float y = s.meta.height / 2 - 0.5;
        float d[3];
        if( orientation[6] != 0 || orientation[7] != 0 || orientation[8] != 0)
        {
            d[0] = originX + y*orientation[3]*pixelSpacingY + x*orientation[0]*pixelSpacingX;
            d[1] = originY + y*orientation[4]*pixelSpacingY + x*orientation[1]*pixelSpacingX;
            d[2] = originZ + y*orientation[5]*pixelSpacingY + x*orientation[2]*pixelSpacingX;
        }
        else
        {
            d[0] = originX + x*pixelSpacingX;
            d[1] = originY + y*pixelSpacingY;
            d[2] = originZ;
        }
        float *centerPix = d;
        float sliceLocation;
        if( fabs( orientation[6]) > fabs(orientation[7]) && fabs( orientation[6]) > fabs(orientation[8]))
            sliceLocation = centerPix[ 0];
       
        if( fabs( orientation[7]) > fabs(orientation[6]) && fabs( orientation[7]) > fabs(orientation[8]))
            sliceLocation = centerPix[ 1];
       
        if( fabs( orientation[8]) > fabs(orientation[6]) && fabs( orientation[8]) > fabs(orientation[7]))
            sliceLocation = centerPix[ 2];
        return sliceLocation;
    }
};


int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string input_dir;
    string output_dir;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    ("output,o", po::value(&output_dir), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);
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

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);
    GlobalInit(argv[0], config);
    Study study(input_dir, true, true, true);
    for (auto const &s: study) {
        cout << s.dir().filename() << '\t' << s[0].meta.slice_location << '\t' << s[0].meta.z << '\t' << OsirisLocation::compute(s[0]) << endl;
    }
    return 0;
}

