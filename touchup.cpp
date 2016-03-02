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

void eval_v (vector<float> const &v1,
             vector<float> const &v2,
             vector<float> *out) {
    CHECK(v1.size() == v2.size());
    out->clear();
    float corr = cv::compareHist(v1, v2, CV_COMP_CORREL);
    out->push_back(corr);
    float err = 0;
    for (unsigned i = 0; i < v1.size(); ++i) {
        float x = v1[i] - v2[i];
        err += x * x;
    }
    err /= v1.size();
    err = sqrt(err);
    out->push_back(err);
}

float top_th;
void patch_top_bottom (StudyReport &rep) {
    for (unsigned xx = 0;; ++xx) {
        vector<SliceReport *> ss;
        for (auto &sax: rep) {
            if (xx < sax.size()) {
                ss.push_back(&sax[xx]);
            }
        }
        if (ss.empty()) break;
        // find first non-top
        for (unsigned i = 0; i < ss.size(); ++i) {
            if (ss[i]->data[SL_TSCORE] < top_th) {
                for (int j = i-1; j >= 0; --j) {
                    if (ss[j]->data[SL_AREA] > ss[j+1]->data[SL_AREA]) {
                        ss[j]->data[SL_AREA] = ss[j+1]->data[SL_AREA];
                    }
                }
                break;
            }
        }


    }
}
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
            if ((gap > 25)) gap = 0;
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
            if ((gap > 25)) gap = 10;
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
    int cohort;
    vector<float> tft;
    vector<float> eft;
    float sys_t, dia_t; // target
    float sys_p, dia_p; // prediction
    float sys_e, dia_e; // error
    float sys1, dia1;
    float sys2, dia2;
    vector<float> sys_v;
    vector<float> dia_v;
    bool good;
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
        vector<float> *ft;
        if (level == 1) {
            target = (mode == 0) ? s.sys_t : s.dia_t;
            ft = &s.tft;
        }
        else if (level == 2) {
            target = (mode == 0) ? fabs(s.sys_t - s.sys_p) : fabs(s.dia_t - s.dia_p);
            ft = &s.eft;
        }
        else CHECK(0);
        bool is_train = (train.empty() || train.count(s.study));
        ostream &os = is_train ? os_train : os_test;
        if (!is_train) ++ntest;
        os << target;
        dump_ft(*ft, os);
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

void run_eval (vector<Sample> &ss, unordered_set<int> const &train) {
    Eval eval;
    int count = 0;
    float sum = 0;
    float dsys = 0;
    float ddia = 0;
    float dall = 0;
    for (auto &s: ss) {
        if (train.size() && train.count(s.study)) continue;
        dsys += sqr(s.sys_t - s.sys_p);
        dall += sqr(s.sys_t - s.sys_p);
        ddia += sqr(s.dia_t - s.dia_p);
        dall += sqr(s.dia_t - s.dia_p);

        float sys_x = eval.score(s.study, 0, s.sys_v);
        float dia_x = eval.score(s.study, 1, s.dia_v);
        sum += sys_x + dia_x;
        cout << s.study << "_Systole" << '\t' << sys_x << '\t' << s.sys_t << '\t' << s.sys_p << '\t' << (s.sys_t - s.sys_p) << '\t' << s.sys_e << endl;
        cout << s.study << "_Diastole" << '\t' << dia_x << '\t' << s.dia_t << '\t' << s.dia_p << '\t' << (s.dia_t - s.dia_p) << '\t' << s.dia_e << endl;
        ++count;
    }
    cout << sum / (count *2) << endl;
    cout << "sys: " << sqrt(dsys / count) << endl;
    cout << "dia: " << sqrt(ddia / count) << endl;
    cout << "all: " << sqrt(dall / count/2) << endl;
}

void run_show (vector<Sample> const &samples) {
    namespace ba = boost::accumulators;
    vector<float> sys_1, sys_2, sys_t;
    vector<float> dia_1, dia_2, dia_t;
    for (auto &s: samples) {
        sys_1.push_back(s.sys1);
        sys_2.push_back(s.sys2);
        sys_t.push_back(s.sys_t);

        dia_1.push_back(s.dia1);
        dia_2.push_back(s.dia2);
        dia_t.push_back(s.dia_t);

        cout << s.study << "_Systole\t";
        cout << (s.sys_t - s.sys1) << '\t' << (s.sys_t - s.sys2) << '\t' << s.sys_t << '\t' << s.sys1 << '\t' << s.sys2 << endl;
        cout << s.study << "_Diastole\t";
        cout << (s.dia_t - s.dia1) << '\t' << (s.dia_t - s.dia2) << '\t' << s.dia_t << '\t' << s.dia1 << '\t' << s.dia2 << endl;
    }
    vector<float> mm;
    cout << "sys";
    eval_v(sys_1, sys_t, &mm);
    for (auto x: mm) {
        cout << "\t" << x;
    }
    /*
    cout << endl;
    cout << "sys2";
    */
    eval_v(sys_2, sys_t, &mm);
    for (auto x: mm) {
        cout << "\t" << x;
    }
    cout << endl;
    cout << "dia";
    eval_v(dia_1, dia_t, &mm);
    for (auto x: mm) {
        cout << "\t" << x;
    }
    /*
    cout << endl;
    cout << "dia2";
    */
    eval_v(dia_2, dia_t, &mm);
    for (auto x: mm) {
        cout << "\t" << x;
    }
    cout << endl;
    sys_t.insert(sys_t.end(), dia_t.begin(), dia_t.end());
    sys_1.insert(sys_1.end(), dia_1.begin(), dia_1.end());
    eval_v(sys_1, sys_t, &mm);
    cout << "all";
    for (auto x: mm) {
        cout << "\t" << x;
    }
    /*
    cout << endl;
    cout << "all2";
    */
    sys_2.insert(sys_2.end(), dia_2.begin(), dia_2.end());
    eval_v(sys_2, sys_t, &mm);
    for (auto x: mm) {
        cout << "\t" << x;
    }
    cout << endl;
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

void split_by_cohort (vector<Sample> const &s, 
                      vector<Sample> *s1,
                      vector<Sample> *s2) {
    s1->clear();
    s2->clear();
    for (auto const &x: s) {
        if (x.cohort == 0) {
            s1->push_back(x);
        }
        else {
            s2->push_back(x);
        }
    }
}


void load_submit_file (fs::path const &path, unordered_map<int, Sample> *data) {
    data->clear();
    fs::ifstream is(path);
    string line;
    getline(is, line);  // first line
    while (getline(is, line)) {
        using namespace boost::algorithm;
        vector<string> ss;
        split(ss, line, is_any_of(",_"), token_compress_on);
        CHECK(ss.size() == Eval::VALUES + 2);
        int n = lexical_cast<int>(ss[0]);
        vector<float> x;
        for (unsigned i = 0; i < Eval::VALUES; ++i) {
            x.push_back(lexical_cast<float>(ss[2 + i]));
        }
        if (ss[1] == "Systole") {
            data->at(n).sys_v.swap(x);
        }
        else if (ss[1] == "Diastole") {
            data->at(n).dia_v.swap(x);
        }
        else CHECK(0);
    }
}

class Xtor {
public:
    ~Xtor () {}
    virtual bool apply (StudyReport const &rep,
                Sample *) const = 0;
    static Xtor *create (string const &name);
};

class ClinicalXtor: public Xtor {
public:
    bool apply (StudyReport const &rep,
                Sample *s) const {
        auto const &front = rep[0][0];
        s->tft.clear();
        s->tft.push_back(front.meta[Meta::SEX]);
        s->tft.push_back(front.meta[Meta::AGE]);
        s->tft.push_back(front.meta.raw_spacing);
        s->eft = s->tft;
        return true;
    }
};

class OneSaxXtor: public Xtor {
public:
    bool apply (StudyReport const &rep,
                Sample *s) const {
        return false;
    }
};


class FullXtor: public Xtor {
public:
    bool apply (StudyReport const &rep,
                Sample *s) const {
        if (rep.size() < 4) return false;
        float sys1, dia1;
        float sys2, dia2;
        compute2(rep, &sys2, &dia2);
        if (!compute1(rep, &sys1, &dia1)) {
            sys1 = sys2;
            dia1 = dia2;
        }
        auto const &front = rep[0][0];
        vector<float> ft{
            front.meta[Meta::SEX], front.meta[Meta::AGE],
            sys1, dia1, sys2, dia2,
        };
        s->sys1 = sys1;
        s->dia1 = dia1;
        s->sys2 = sys2;
        s->dia2 = dia2;
        s->tft = ft;
        s->eft = ft;
        return true;
    }
};

Xtor *Xtor::create (string const &name) {
    if (name == "cli") return new ClinicalXtor;
    if (name == "one") return new OneSaxXtor;
    if (name == "full") return new FullXtor;
    CHECK(0);
    return nullptr;
}

void preprocess (StudyReport *rep, bool detail, bool top) {
    if (top) {
        patch_top_bottom(*rep);
    }
    if (detail) {
        for (auto &s: rep->back()) {
            s.data[SL_AREA] = 0;
        }
    }
}

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    vector<int> studies;
    string train_path;  // only use IDs in this file for training
                        // exclude these for validation
                        // do not affect show & submit
    string method;
    string xtor_name;
    fs::path root;      // working directory
    fs::path data_root; // report directory
    fs::path raw_root;  // raw data directory
    fs::path buddy;
    fs::path fallback_path;
    int round1, round2;
    float scale;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&studies), "report files")
    ("scale,s", po::value(&scale)->default_value(1.2), "")
    ("keep-tail", "")
    ("method", po::value(&method), "")
    ("train", po::value(&train_path), "limit training in these study IDs")
    ("round1", po::value(&round1)->default_value(2000), "")
    ("round2", po::value(&round2)->default_value(1500), "")
    ("shuffle", "")
    ("data", po::value(&data_root), "dir containing report files")
    ("raw", po::value(&raw_root), "dir containing raw images")
    ("ws,w", po::value(&root), "working directory")
    ("fallback", po::value(&fallback_path), "")
    ("xtor", po::value(&xtor_name)->default_value("full"), "")
    ("cohort", "")
    ("buddy", po::value(&buddy), "")
    ;

    po::positional_options_description p;
    p.add("data", 1);
    p.add("ws", 1);
    p.add("method", 1);
    p.add("input", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || root.empty() || method.empty()) {
        cerr << "ADSB2 VERSION: " << VERSION << endl;
        cerr << desc;
        return 1;
    }
    bool do_detail = !(vm.count("keep-tail") > 0);
    bool do_top = (vm.count("top") > 0);
    bool do_cohort = vm.count("cohort") > 0;

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
    config.put("adsb2.models", root.native()); // find models in workspace

    GlobalInit(argv[0], config);

    // load training set
    unordered_set<int> train_set;
    if (train_path.size()) {
        int x;
        ifstream is(train_path.c_str());
        while (is >> x) {
            train_set.insert(x);
        }
    }

    // load fallback data
    unordered_map<int, Sample> fallback;
    if (!fallback_path.empty()) {
        load_submit_file(fallback_path, &fallback);
    }

    // load cohort data
    unordered_map<int, int> cohort;
    if (do_cohort) {
        int id, c;
        fs::ifstream is(home_dir/fs::path("cohort"));
        while (is >> id >> c) {
            cohort[id] = c;
        }
    }

    Eval eval;
    Xtor *xtor = Xtor::create(xtor_name);
    CHECK(xtor);
    vector<Sample> samples;
    int level = 0;
    //  level 0:    no model
    //  level 1:    target model
    //  level 2:    target model & error model
    if (method == "show") level = 0;
    else if (method == "train1") level = 0;
    else if (method == "train2") level = 1;
    else if (method == "eval") level = 2;
    else if (method == "submit") level = 2;
    else CHECK(0) << "method " << method << " not supported";

    Classifier *target_sys[2] = {0, 0};
    Classifier *target_dia[2] = {0, 0};
    Classifier *error_sys = 0;
    Classifier *error_dia = 0;
    if (level >= 1) {   // target model
        target_sys[0] = Classifier::get("target.sys.0");
        target_dia[0] = Classifier::get("target.dia.0");
        if (do_cohort) {
            target_sys[1] = Classifier::get("target.sys.1");
            target_dia[1] = Classifier::get("target.dia.1");
        }
    }
    if (level >= 2) {   // error model
        error_sys = Classifier::get("error.sys");
        error_dia = Classifier::get("error.dia");
    }

    fs::create_directories(root);
    for (auto const &study: studies) {
        Sample s;
        s.study = study;
        s.good = true;
        s.cohort = 0;

        fs::path path(data_root / fs::path(lexical_cast<string>(study))/ fs::path("report.txt"));
        StudyReport x(path);
        if (do_cohort) {
            if (cohort.size()) {
                auto it = cohort.find(s.study);
                CHECK(it != cohort.end());
                s.cohort = it->second;
            }
            else if (x.size() && x[0].size()) {
                s.cohort = x[0][0].data[SL_COHORT];
            }
        }
        if (x.empty()) {
            LOG(ERROR) << "Fail to load data file: " << path.native();
            s.good = false;
            // probe ...
        }
        preprocess(&x, do_detail, do_top);
        s.good = s.good && xtor->apply(x, &s);
        if (!buddy.empty()) {
#if 0
            fs::path buddy_path = buddy / fs::path(lexical_cast<string>(s.study)) / fs::path("report.txt");
            StudyReport bx(buddy_path);
            if (bx.empty()) {
                LOG(ERROR) << "Fail to load data file: " << buddy_path.native();
            }
            preprocess(&bx, do_detail, do_top);
            Sample bs;
            s.good = s.good && xtor->apply(bx, &bs);
            // 0 1 2 3     4   5
            //       dia1      dia2
            s.tft[3] = bs.tft[3];
            s.tft[5] = bs.tft[5];
#endif
        }

        s.sys_t = eval.get(s.study, 0);
        s.dia_t = eval.get(s.study, 1);
        if (level >= 1) {
            s.sys_p = target_sys[s.cohort]->apply(s.tft);
            s.dia_p = target_dia[s.cohort]->apply(s.tft);
        }
        if (level >= 2) {
            s.sys_e = error_sys->apply(s.eft) * scale;
            s.dia_e = error_dia->apply(s.eft) * scale;
            if (s.good) {
                GaussianAcc(s.sys_p, s.sys_e, &s.sys_v);
                GaussianAcc(s.dia_p, s.dia_e, &s.dia_v);
            }
            else {
                LOG(WARNING) << "Study " << s.study << " not good, using fallback";
                auto it = fallback.find(s.study);
                if (it == fallback.end()) {
                    LOG(WARNING) << "Cannot find fallback for study " << s.study;
                }
                else {
                    if (it->second.sys_v.size() == Eval::VALUES) {
                        s.sys_v = it->second.sys_v;
                    }
                    else {
                        LOG(WARNING) << "Cannot find sys fallback for study, you are doomed " << s.study;
                        GaussianAcc(200, 100, &s.sys_v);
                    }
                    if (it->second.dia_v.size() == Eval::VALUES) {
                        s.dia_v = it->second.dia_v;
                    }
                    else {
                        LOG(WARNING) << "Cannot find dia fallback for study, you are doomed " << s.study;
                        GaussianAcc(400, 100, &s.sys_v);
                    }
                }
            }
        }
        samples.push_back(s);
    }

    if (vm.count("shuffle")){
        random_shuffle(samples.begin(), samples.end());
    }

    if (method == "show") {
        run_show(samples);
    }
    if (method == "train1") {
        vector<Sample> c0, c1;
        if (do_cohort) {
            split_by_cohort(samples, &c0, &c1);
        }
        else {
            c0 = samples;
        }
        run_train(c0, 1, 0, root/fs::path("d.target.sys.0"), train_set, root/fs::path("target.sys.0"), round1);
        run_train(c0, 1, 1, root/fs::path("d.target.dia.0"), train_set, root/fs::path("target.dia.0"), round1);
        if (do_cohort) {
            run_train(c1, 1, 0, root/fs::path("d.target.sys.1"), train_set, root/fs::path("target.sys.1"), round1);
            run_train(c1, 1, 1, root/fs::path("d.target.dia.1"), train_set, root/fs::path("target.dia.1"), round1);
        }
    }
    else if (method == "train2") {
        run_train(samples, 2, 0, root/fs::path("d.error.sys"), train_set, root/fs::path("error.sys"), round2);
        run_train(samples, 2, 1, root/fs::path("d.error.dia"), train_set, root/fs::path("error.dia"), round2);
    }
    else if (method == "eval") {
        run_eval(samples, train_set);
    }
    else if (method == "submit") {
        run_submit(samples);
    }
    return 0;
}

