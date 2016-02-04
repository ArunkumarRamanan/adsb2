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
    string input_dir;
    string output_dir;
    /*
    string output_dir;
    string gif;
    float th;
    int mk;
    */

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&input_dir), "")
    ("output,o", po::value(&output_dir), "")
    ("bound", "")
    ("no-gif", "")
    //("output,o", po::value(&output_dir), "")
    /*
    ("gif", po::value(&gif), "")
    ("th", po::value(&th)->default_value(0.90), "")
    ("mk", po::value(&mk)->default_value(5), "")
    */
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

    bool do_gif = vm.count("no-gif") == 0;

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);
    Cook cook(config);

    timer::auto_cpu_timer timer(cerr);
    Study study(input_dir, true, true, true);
    cook.apply(&study);
    cv::Rect bound;
    string bound_model = config.get("adsb2.caffe.bound_model", (home_dir/fs::path("bound_model")).native());
    if (vm.count("bound")) {
        Detector *bb_det = make_caffe_detector(bound_model);
        Bound(bb_det, &study, &bound, config);
        delete bb_det;
    }

    ComputeBoundProb(&study, config);
    cerr << "Filtering..." << endl;
    ProbFilter(&study, config);
    /*
    for (auto &s: study) {
        MotionFilter(&s, config);
    }
    */
    cerr << "Finding squares..." << endl;
#pragma omp parallel for schedule(dynamic, 1)
    for (unsigned i = 0; i < slices.size(); ++i) {
        FindSquare(slices[i]->prob,
                  &slices[i]->pred_box, config);
    }
    ComputeContourProb(&study, config);
    study_CA1(&study, config);
    Volume min, max;
    FindMinMaxVol(study, &min, &max, config);
    if (output_dir.size()) {
        cerr << "Saving output..." << endl;
        fs::path dir(output_dir);
        fs::create_directories(dir);
        {
            fs::ofstream vol(dir/fs::path("volume.txt"));
            vol << min.mean << '\t' << std::sqrt(min.var)
                << '\t' << max.mean << '\t' << std::sqrt(max.var) << endl;
        }
        fs::ofstream html(dir/fs::path("index.html"));
        html << "<html><body>" << endl;
        html << "<table border=\"1\"><tr><th>Study</th><th>Sex</th><th>Age</th></tr>"
             << "<tr><td>" << study.dir().native() << "</td><td>" << (study.front().front().meta[Meta::SEX] ? "Female": "Male")
             << "</td><td>" << study.front().front().meta[Meta::AGE]
             << "</td></tr></table>" << endl;
        html << "<br/><img src=\"radius.png\"></img>" << endl;
        html << "<br/><table border=\"1\">"<< endl;
        html << "<tr><th>Slice</th><th>Location</th><th>Thickness</th><th>Interval</th><th>image</th></tr>";
        fs::path gp1(dir/fs::path("plot.gp"));
        fs::ofstream gp(gp1);
        gp << "set xlabel \"time\";" << endl;
        gp << "set ylabel \"location\";" << endl;
        gp << "set zlabel \"radius\";" << endl;
        gp << "set hidden3d;" << endl;
        gp << "set style data pm3d;" << endl;
        gp << "set dgrid3d 50,50 qnorm 2;" << endl;
        gp << "splot '-' using 1:2:3 notitle" << endl;
        if (do_gif) {
#pragma omp parallel for
            for (unsigned i = 0; i < study.size(); ++i) {
                study[i].visualize();
                study[i].save_gif(dir/fs::path(fmt::format("{}.gif", i)));
            }
        }
        for (unsigned i = 0; i < study.size(); ++i) {
            html << "<tr>"
                 << "<td>" << study[i].dir().filename().native() << "</td>"
                 << "<td>" << study[i].front().meta[Meta::SLICE_LOCATION] << "</td>"
                 << "<td>" << study[i].front().meta[Meta::SLICE_THICKNESS] << "</td>"
                 << "<td>" << study[i].front().meta[Meta::NOMINAL_INTERVAL] << "</td>"
                 << "<td><img src=\"" << i << ".gif\"></img></td></tr>" << endl;
            for (auto const &s: study[i]) {
                float r = std::sqrt(s.pred_box.area())/2 * s.meta.spacing;
                gp << s.meta.trigger_time
                   << '\t' << s.meta[Meta::SLICE_LOCATION]
                   << '\t' << r << endl;
            }
        }
        gp << 'e' << endl;
        html << "</table></body></html>" << endl;
        fs::ofstream os(dir/fs::path("report.txt"));
        for (auto const &series: study) {
            for (auto const &s: series) {
                report(os, s, bound);
            }
        }
        {
            fs::path gp2(dir/fs::path("plot2.gp"));
            fs::ofstream gp(gp2);
            gp << "set terminal png;" << endl;
            gp << "set output \"" << (dir/fs::path("radius.png")).native() << "\";" << endl;
            gp << "load \"" << gp1.native() << "\";" << endl;
            gp.close();
            string cmd = fmt::format("gnuplot {}", gp2.string());
            ::system(cmd.c_str());
            fs::remove(gp2);
        }
    }
    else {
        for (auto const &series: study) {
            for (auto const &s: series) {
                report(cout, s, bound);
            }
        }
    }
    return 0;
}

