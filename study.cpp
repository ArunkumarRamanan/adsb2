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

map<string, vector<pair<string, float>>> presets = {
    {"sys", {
        {"adsb2.ca1.margin2", 40},
        {"adsb2.ca1.th1", 0.80},
        {"adsb2.ca1.th2", 0.05},
        {"adsb2.ca1.smooth1", 10},
        {"adsb2.ca1.smooth2", 255},
        {"adsb2.ca1.ndisc", 0.2},
        {"adsb2.ca1.wctrpct", 0.9},
        {"adsb2.ca1.ctrpct", 0.8},
        {"adsb2.ca1.gth2", 1}
    }},
    {"dia", {
        //{"adsb2.ca1.margin2", 40},
        {"adsb2.ca1.th1", 0.83},
        /*
        {"adsb2.ca1.th2", 0.05},
        {"adsb2.ca1.smooth1", 10},
        {"adsb2.ca1.smooth2", 255},
        {"adsb2.ca1.ndisc", 0.2},
        {"adsb2.ca1.wctrpct", 0.9},
        {"adsb2.ca1.ctrpct", 0.8}
        */
    }}
};

int main(int argc, char **argv) {
    nice(10);
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    fs::path input_path;
    fs::path dir;
    fs::path snapshot_path;
    int ca;
    string preset;
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
    ("input,i", po::value(&input_path), "")
    ("snapshot,s", "input is snapshot")
    ("os", po::value(&snapshot_path), "")
    ("output,o", po::value(&dir), "")
    ("ca", po::value(&ca)->default_value(1), "")
    ("bound", "")
    ("top", "")
    ("bottom", "")
    ("preset", po::value(&preset), "")
    ("gif", "")
    ("gnuplot", "")
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

    if (vm.count("help") || input_path.empty()) {
        cerr << "ADSB2 VERSION: " << VERSION << endl;
        cerr << desc;
        return 1;
    }

    bool do_gif = vm.count("gif") > 0;

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }

    if (preset.size()) {
        auto it = presets.find(preset);
        CHECK(it != presets.end()) << "preset " << preset << " not found.";
        for (auto const &p: it->second) {
            config.put(p.first, p.second);
        }
    }

    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);
    Cook cook(config);

    timer::auto_cpu_timer timer(cerr);
    Study study;
    if (vm.count("snapshot")) {
        study.load(input_path);
    }
    else {
        study.load_raw(input_path, true, true, true);
        cook.apply(&study);
        cv::Rect bound;
        /*
        string bound_model = config.get("adsb2.caffe.bound_model", (home_dir/fs::path("bound_model")).native());
        if (vm.count("bound")) {
            Detector *bb_det = make_caffe_detector(bound_model);
            Bound(bb_det, &study, &bound, config);
            delete bb_det;
        }
        */
        vector<Slice *> slices;
        study.pool(&slices);
        if (vm.count("top")) {
            ComputeTop(&study, config);
        }
#ifdef USE_TOP
        for (auto &ss: study) {
            float sum = 0;
            for (auto &s: ss) {
                sum += s.data[SL_TSCORE];
            }
            sum /= ss.size();
            if (sum > 0.8) {
                for (auto &s: ss) {
                    s.images[IM_IMAGE2] = s.images[IM_IMAGE];
                    s.images[IM_IMAGE] = cv::Mat();
                }
            }
        }
#endif
        ComputeBoundProb(&study);
#ifdef USE_TOP
        ApplyDetector("top_bound", &study, IM_IMAGE2, IM_PROB2, 1.0, 0);
        for (Slice *s: slices) {
            if (s->images[IM_PROB2].data) {
                s->images[IM_PROB] = s->images[IM_PROB2];
                s->images[IM_IMAGE] = s->images[IM_IMAGE2];
            }
        }
#endif
        cerr << "Filtering..." << endl;
        ProbFilter(&study, config);
        cerr << "Finding squares..." << endl;
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < slices.size(); ++i) {
            FindBox(slices[i], config);
        }

        ComputeContourProb(&study, config);
    }
    study_CA1(&study, config, true);
    if (vm.count("bottom")) {
        EvalBottom(&study, config);
        RefineBottom(&study, config);
    }
#if 0
    if (decap > 0) {
        CHECK(decap < 5);
        for (int i = 0; i < decap; ++i) {
            for (auto &s: study[i]) {
                s.area = 0;
            }
        }
    }
    else if (decap < 0) {
        RefineTop(&study, config);
    }
    TrimBottom(&study, config);
#endif
    
    Volume min, max;
    FindMinMaxVol(study, &min, &max, config);
    if (!snapshot_path.empty()) {
        fs::path parent = snapshot_path.parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent);
        }
        study.save(snapshot_path);
    }
    if (!dir.empty()) {
        cerr << "Saving output..." << endl;
        fs::create_directories(dir);
        {
            fs::ofstream vol(dir/fs::path("volume.txt"));
            vol << min.mean << '\t' << std::sqrt(min.var)
                << '\t' << max.mean << '\t' << std::sqrt(max.var) << endl;
        }
        {
            fs::ofstream vol(dir/fs::path("coef.txt"));
            vol << min.mean << '\t' << min.coef1 << '\t' << min.coef2
                << '\t' << max.mean << '\t' << max.coef1 << '\t' << max.coef2 << endl;
        }
        fs::ofstream html(dir/fs::path("index.html"));
        html << "<html><body>" << endl;
        html << "<table border=\"1\"><tr><th>Study</th><th>Sex</th><th>Age</th></tr>"
             << "<tr><td>" << study.dir().native() << "</td><td>" << (study.front().front().meta[Meta::SEX] ? "Female": "Male")
             << "</td><td>" << study.front().front().meta[Meta::AGE]
             << "</td></tr></table>" << endl;
        html << "<br/><img src=\"radius.png\"></img>" << endl;
        html << "<br/><table border=\"1\">"<< endl;
        html << "<tr><th>Slice</th><th>Location</th><th>image</th></tr>";
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
            auto &ss = study[i];
            html << "<tr>"
                 << "<td>" << study[i].dir().filename().native() << "</td>"
                 << "<td>" << study[i].front().meta.slice_location << "</td>"
            //     << "<td>" << study[i].front().meta[Meta::NOMINAL_INTERVAL] << "</td>"
                 << "<td><img src=\"" << i << ".gif\"></img></td></tr>" << endl;
        }
        gp << 'e' << endl;
        html << "</table></body></html>" << endl;
        fs::ofstream os(dir/fs::path("report.txt"));
        for (unsigned i = 0; i < study.size(); ++i) {
            auto const &series = study[i];
            for (unsigned j = 0; j < series.size(); ++j) {
                auto const &s = series[j];
                os << s.path.native()
                    << '\t' << i
                    << '\t' << j
                    << '\t' << s.data[SL_AREA]
                    << '\t' << s.box.x
                    << '\t' << s.box.y
                    << '\t' << s.box.width
                    << '\t' << s.box.height
                    << '\t' << s.polar_box.x
                    << '\t' << s.polar_box.y
                    << '\t' << s.polar_box.width
                    << '\t' << s.polar_box.height
                    << '\t' << s.meta.slice_location
                    << '\t' << s.meta.trigger_time
                    << '\t' << s.meta.spacing
                    << '\t' << s.meta.raw_spacing;
                for (auto const &v: s.meta) {
                    os << '\t' << v;
                }
                for (auto const &v: s.data) {
                    os << '\t' << v;
                }
                os << std::endl;
            }
        }
        if (vm.count("gnuplot")) {
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
    /*
    else {
        for (auto const &series: study) {
            for (auto const &s: series) {
                report(cout, s, bound);
            }
        }
    }
    */
    return 0;
}

