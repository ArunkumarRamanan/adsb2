#include <sstream>
#include <unordered_set>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "adsb2.h"

namespace adsb2 {
    extern fs::path home_dir;
}

using namespace std;
using namespace adsb2;

// models:
//      sys
//      dia
//      sys_err
//      dia_err
// each model:
//      train set
//      test set
//      script
//

void join (vector<string> const &v, string *acc) {
    ostringstream os;
    for (unsigned i = 0; i < v.size(); ++i) {
        if (i) os << ' ';
        os << v[i];
    }
    *acc = os.str();
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

void run_train (vector<Sample> &ss, int level, int mode, fs::path const &dir, unordered_set<int> const &train,
        fs::path const &model, int round) {
    fs::create_directories(dir);
    CHECK(mode == 0 || mode == 1);
    CHECK(level == 1 || level == 2);
    fs::path train_path(dir/fs::path("train"));
    fs::path test_path(dir/fs::path("test"));

    fs::ofstream os_train(train_path);
    fs::ofstream os_test(test_path);
    int ntest = 0;
    for (auto &s: ss) {
        float target;
        if (level == 1) {
            target = (mode == 0) ? s.sys_t : s.dia_t;
        }
        else if (level == 2) {
            target = (mode == 0) ? fabs(s.sys_t - s.sys_p) : fabs(s.dia_t - s.dia_p);
        }
        else CHECK(0);
        bool is_train = (train.empty() || train.count(s.study));
        ostream &os = is_train ? os_train : os_test;
        if (!is_train) ++ntest;
        os << target;
        dump_ft(s.ft, os);
        os << endl;
    }
    if (!model.empty()) {
        vector<string> cmd;
        cmd.push_back((home_dir/fs::path("data/xgboost")).native());
        cmd.push_back((home_dir/fs::path("data/xglinear.conf")).native());
        cmd.push_back(fmt::format("data={}", train_path.native()));
        if (ntest) {
            cmd.push_back(fmt::format("eval[test]={}", test_path.native()));
        }
        cmd.push_back(fmt::format("num_round={}", round));
        cmd.push_back(fmt::format("model_out={}", model.native()));
        cmd.push_back("2>&1");
        cmd.push_back("|");
        cmd.push_back("tee");
        cmd.push_back((dir/fs::path("log")).native());
        string x;
        join(cmd, &x);
        fs::ofstream os(dir/fs::path("cmd"));
        os << x << endl;
        ::system(x.c_str());
    }
}

void run_eval (vector<Sample> &ss) {
    Eval eval;
    float sum = 0;
    for (auto &s: ss) {
        float sys_x = eval.score(s.study, 0, s.sys_v);
        float dia_x = eval.score(s.study, 1, s.dia_v);
        sum += sys_x + dia_x;
        cout << s.study << "_Systole" << '\t' << sys_x << '\t' << s.sys_t << '\t' << s.sys_p << '\t' << (s.sys_t - s.sys_p) << '\t' << s.sys_e << endl;
        cout << s.study << "_Diastole" << '\t' << dia_x << '\t' << s.dia_t << '\t' << s.dia_p << '\t' << (s.dia_t - s.dia_p) << '\t' << s.dia_e << endl;
    }
    cout << sum / (ss.size() *2) << endl;
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
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    vector<string> paths;
    string train_path;
    string cohort_path;
    string method;
    string ws;
    float scale;
    int round1, round2;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&paths), "")
    ("scale,s", po::value(&scale)->default_value(1.2), "")
    ("detail", "")
    ("method", po::value(&method), "")
    ("train", po::value(&train_path), "")
    ("cohort", po::value(&cohort_path), "")
    ("round1", po::value(&round1)->default_value(2000), "")
    ("round2", po::value(&round2)->default_value(1500), "")
    ("shuffle", "")
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
    config.put("adsb2.models", ws); // find models in workspace

    GlobalInit(argv[0], config);
    unordered_set<int> train_set;

    if (train_path.size()) {
        int x;
        ifstream is(train_path.c_str());
        while (is >> x) {
            train_set.insert(x);
        }
    }

    unordered_map<int, int> cohort;
    if (cohort_path.size()) {
        int id, c;
        ifstream is(cohort_path.c_str());
        while (is >> id >> c) {
            cohort[id] = c;
        }
    }

    Eval eval;
    vector<Sample> samples;
    int level = 0;
    //  level 0:    no model
    //  level 1:    target model
    //  level 2:    target model & error model
    if (method == "train1") level = 0;
    else if (method == "train2") level = 1;
    else if (method == "eval") level = 2;
    else if (method == "submit") level = 2;
    else CHECK(0) << "method " << method << " not supported";

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

    fs::path root(ws);
    fs::create_directories(root);
    for (auto const &p: paths) {
        fs::path path(p);
        StudyReport x(path);
        if (x.empty()) {
            LOG(ERROR) << "Cannot load " << path;
            continue;
        }
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
        auto &front = x[0][0];
        vector<float> ft{sys1, dia1, sys2, dia2, front.meta[Meta::SEX], front.meta[Meta::AGE], front.meta[Meta::SLICE_THICKNESS], front.meta.raw_spacing}; //, front.meta.PercentPhaseFieldOfView};
        if (cohort.size()) {
            auto it = cohort.find(s.study);
            CHECK(it != cohort.end());
            if (it->second) {
                ft.push_back(0);
                ft.push_back(1);
            }
            else {
                ft.push_back(1);
                ft.push_back(0);
            }
        }
        s.ft = ft;
        s.sys_t = eval.get(s.study, 0);
        s.dia_t = eval.get(s.study, 1);
        if (level >= 1) {
            s.sys_p = target_sys->apply(s.ft);
            s.dia_p = target_dia->apply(s.ft);
        }
        if (level >= 2) {
            s.sys_e = error_sys->apply(s.ft) * scale;
            s.dia_e = error_dia->apply(s.ft) * scale;
            GaussianAcc(s.sys_p, s.sys_e, &s.sys_v);
            GaussianAcc(s.dia_p, s.dia_e, &s.dia_v);
        }
        samples.push_back(s);
    }

    if (vm.count("shuffle")){
        random_shuffle(samples.begin(), samples.end());
    }

    if (method == "train1") {
        run_train(samples, 1, 0, root/fs::path("d.target.sys"), train_set, root/fs::path("target.sys"), round1);
        run_train(samples, 1, 1, root/fs::path("d.target.dia"), train_set, root/fs::path("target.dia"), round1);
    }
    else if (method == "train2") {
        run_train(samples, 2, 0, root/fs::path("d.error.sys"), train_set, root/fs::path("error.sys"), round2);
        run_train(samples, 2, 1, root/fs::path("d.error.dia"), train_set, root/fs::path("error.dia"), round2);
    }
    else if (method == "eval") {
        run_eval(samples);
    }
    else if (method == "submit") {
        run_submit(samples);
    }
    return 0;
}
