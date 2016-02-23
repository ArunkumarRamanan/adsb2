#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "adsb2.h"

using namespace std;
using namespace adsb2;

static inline float sqr (float x) {
    return x * x;
}

void GaussianAcc (float v, float scale, vector<float> *s) {
    s->resize(Eval::VALUES);
    float sum = 0;
    for (unsigned i = 0; i < Eval::VALUES; ++i) {
        float x = (float(i) - v) / scale;
        x = exp(-0.5 * x * x);
        s->at(i) = x;
        sum += x;
    }
    float acc = 0;
    for (auto &v: *s) {
        acc += v;
        v = acc / sum;
    }
}

void compute (StudyReport const &rep, float *sys, float *dia, float *dgap) {
    vector<float> all(rep[0].size(), 0);
    for (unsigned sl = 0; sl < all.size(); ++sl) {
        float v = 0;
        for (unsigned sax = 0; sax + 1 < rep.size(); ++sax) {

            float a = rep[sax][sl].data[SL_AREA] * sqr(rep[sax][sl].meta.spacing);
            float b = rep[sax+1][sl].data[SL_AREA] * sqr(rep[sax+1][sl].meta.spacing);
            float gap = fabs(rep[sax+1][sl].meta.slice_location
                      - rep[sax][sl].meta.slice_location);
            v += (a + b + sqrt(a*b)) * gap / 3;
        }
        all[sl] = v / 1000;
    }
    float m = all[0];
    float M = all[0];
    for (unsigned i = 1; i < all.size(); ++i) {
        if (all[i] < m) m = all[i];
        if (all[i] > M) M = all[i];
    }
    *dia = M;
    *sys = m;
    *dgap = 0;
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
        float sys, dia, dgap;
        float gs_sys, gs_dia;
        int study = x.front().front().study_id;
        if (detail) {
            for (auto &s: x.back()) {
                s.data[SL_AREA] = 0;
            }
        }
        compute(x, &sys, &dia, &dgap);
        gs_sys = eval.get(study, 0);
        gs_dia = eval.get(study, 1);
        float dsys = gs_sys - sys;
        float ddia = gs_dia - dia;
        GaussianAcc(sys, scale, &v);
        cout << study << "_Systole" << '\t' << eval.score(study, 0, v) << '\t' << dsys << '\t' << gs_sys << '\t' << sys << '\t' << (dsys-ddia) << '\t' << dgap << endl;
        GaussianAcc(dia, scale, &v);
        cout << study << "_Diastol" << '\t' << eval.score(study, 1, v) << '\t' << ddia << '\t' << gs_dia << '\t' << dia << '\t' << (dsys-ddia) << '\t' << dgap << endl;
    }
}

