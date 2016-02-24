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

bool compute1 (StudyReport const &rep, float *sys, float *dia) {
    vector<float> all(rep[0].size(), 0);
    for (unsigned i = 1; i < rep.size(); ++i) {
        if (rep[i].size() != rep[0].size()) return false;
    }
    for (unsigned sl = 0; sl < all.size(); ++sl) {
        float v = 0;
        for (unsigned sax = 0; sax + 1 < rep.size(); ++sax) {

            float a = rep[sax][sl].data[SL_AREA] * sqr(rep[sax][sl].meta.spacing);
            float b = rep[sax+1][sl].data[SL_AREA] * sqr(rep[sax+1][sl].meta.spacing);
            float gap = fabs(rep[sax+1][sl].meta.slice_location
                      - rep[sax][sl].meta.slice_location);
            if (gap > 25) gap = 10;
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
    if (M < all[0] * 1.2) M = all[0];
    *dia = M;
    *sys = m;
    return true;
}

void compute2 (StudyReport const &rep, float *sys, float *dia) {
    float oma = 0, oMa = 0, ol = 0;
    float m = 0, M = 0;
    for (unsigned i = 0; i < rep.size(); ++i) {
        auto const &ss = rep[i];
        float ma = ss[0].data[SL_AREA] * sqr(ss[0].meta.spacing);
        float Ma = ma;
        for (auto const &s: ss) {
            float x = s.data[SL_AREA] * sqr(s.meta.spacing);
            if (x < ma) ma = x;
            if (x > Ma) Ma = x;
        }
        float l = ss[0].meta.slice_location;
        if (i > 0) {
            float gap = abs(ol - l);
            if (gap > 25) gap = 10;
            m += (oma + ma + sqrt(oma * ma)) * gap/3;
            M += (oMa + Ma + sqrt(oMa * Ma)) * gap/3;
        }
        oma = ma;
        oMa = Ma;
        ol = l;
    }
    *sys = m/1000;
    *dia = M/1000;
}

void dump_ft (vector<float> const &ft, ostream &os) {
    for (unsigned i = 0; i < ft.size(); ++i) {
        os << " " << i << ":" << ft[i];
    }
}

struct Sample {
    int study;
    vector<float> ft;
    float sys_t, dia_t; // target
    float sys_p, dia_p; // prediction
    float sys_e, dia_e; // error
    vector<float> sys_v;
    vector<float> dia_v;
};

void run_train1 (vector<Sample> &ss, int mode) {
    CHECK(mode >= 1 && mode <= 3);
    Eval eval;
    for (auto &s: ss) {
        if (mode & 1) {
            cout << s.sys_t;
            dump_ft(s.ft, cout);
            cout << endl;
        }
        if (mode & 2) {
            cout << s.dia_t;
            dump_ft(s.ft, cout);
            cout << endl;
        }
    }
}

void run_train2 (vector<Sample> &ss, int mode) {
    CHECK(mode >= 1 && mode <= 3);
    Eval eval;
    for (auto &s: ss) {
        if (mode & 1) {
            cout << fabs(s.sys_t - s.sys_p);
            dump_ft(s.ft, cout);
            cout << endl;
        }
        if (mode & 2) {
            cout << fabs(s.dia_t - s.dia_p);
            dump_ft(s.ft, cout);
            cout << endl;
        }
    }
}

void run_eval (vector<Sample> &ss) {
    Eval eval;
    float sum = 0;
    for (auto &s: ss) {
        float sys_x = eval.score(s.study, 0, s.sys_v);
        float dia_x = eval.score(s.study, 1, s.dia_v);
        sum += sys_x + dia_x;
        cout << s.study << "_Systole" << '\t' << sys_x << endl;
        cout << s.study << "_Diastole" << '\t' << dia_x << endl;
    }
    cout << sum / (ss.size() *2);
}

void run_submit (vector<Sample> &ss) {
    cout << adsb2::HEADER << endl;
    for (auto &s: ss) {
        cout << s.study << "_Systole";
        for (auto &f: s.sys_v) {
            cout << ',' << f;
        }
        cout << endl;
        cout << s.study << "_Diastole";
        for (auto &f: s.dia_v) {
            cout << ',' << f;
        }
        cout << endl;
    }
}


int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    vector<string> paths;
    string method;
    string ws;
    float scale;
    int mode;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&paths), "")
    ("scale,s", po::value(&scale)->default_value(20), "")
    ("detail", "")
    ("method", po::value(&method), "")
    ("mode", po::value(&mode)->default_value(1), "")
    ("ws,w", po::value(&ws), "")
    ;

    po::positional_options_description p;
    p.add("ws", 1);
    p.add("method", 1);
    p.add("input", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || ws.empty() || method.empty()) {
        cerr << desc;
        return 1;
    }
    bool detail = vm.count("detail") > 0;
    if (paths.empty()) {
        string p;
        while (cin >> p) {
            paths.push_back(p);
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
    Eval eval;
    vector<Sample> samples;
    int level = 0;
    if (method == "train1") level = 0;
    if (method == "train2") level = 1;
    if (method == "eval") level = 2;
    if (method == "submit") level = 2;

    Classifier *target_sys = 0;
    Classifier *target_dia = 0;
    Classifier *error_sys = 0;
    Classifier *error_dia = 0;
    if (level >= 1) {
        target_sys = Classifier::get("target.sys");
        target_dia = Classifier::get("target.dia");
    }
    if (level >= 2) {
        error_sys = Classifier::get("error.sys");
        error_dia = Classifier::get("error.dia");
    }
    for (auto const &p: paths) {
        fs::path path(p);
        StudyReport x(path);
        Sample s;
        float sys1, dia1;
        float sys2, dia2;
        s.study = x.front().front().study_id;
        if (detail) {
            for (auto &s: x.back()) {
                s.data[SL_AREA] = 0;
            }
        }
        compute2(x, &sys2, &dia2);
        if (!compute1(x, &sys1, &dia1)) {
            sys1 = sys2;
            dia1 = dia2;
        }
        vector<float> ft{sys1, dia1, sys2, dia2, x[0][0].meta[Meta::SEX], x[0][0].meta[Meta::AGE]};
        s.ft = ft;
        s.sys_t = eval.get(s.study, 0);
        s.dia_t = eval.get(s.study, 1);
        if (level >= 1) {
            s.sys_p = target_sys->apply(s.ft);
            s.dia_p = target_dia->apply(s.ft);
        }
        if (level >= 2) {
            s.sys_e = error_sys->apply(s.ft);
            s.dia_e = error_dia->apply(s.ft);
            GaussianAcc(s.sys_p, s.sys_e, &s.sys_v);
            GaussianAcc(s.dia_p, s.dia_e, &s.dia_v);
        }
        samples.push_back(s);
    }

    if (method == "train1") {
        run_train1(samples, mode);
    }
    else if (method == "train2") {
        run_train2(samples, mode);
    }
    else if (method == "eval") {
        run_eval(samples);
    }
    else if (method == "submit") {
        run_submit(samples);
    }
    return 0;
}

