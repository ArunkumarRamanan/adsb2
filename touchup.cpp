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

void Smooth (StudyReport *study, Config const &conf);

struct Fallback {
    bool found;
    float sys_p, dia_p; // prediction
    float sys_e, dia_e; // error

    float sys_p_raw, dia_p_raw; // prediction
    float sys_e_raw, dia_e_raw; // error
};

Fallback global_fb = { true, 50, 50, 100, 50};

struct Sample {
    int study;
    bool good;          // if not good, DO not use for training
                        // and use fallback's prediction
    int cohort;
    float age;
    vector<float> tft_sys;  // target feature
    vector<float> tft_dia;  // target feature
    vector<float> eft;  // error feature

    float sys_t, dia_t; // target
    float sys_p, dia_p; // prediction
    float sys_e, dia_e; // error
    float sys_p_raw, dia_p_raw; // prediction
    float sys_e_raw, dia_e_raw; // error
    vector<float> sys_v;
    vector<float> dia_v;

    // only using full extractor
    float sys1, dia1;   // computed with method1
    float sys2, dia2;   // computed with method2
    Fallback fb;
    Fallback fb2;
};


class Xtor {
public:
    ~Xtor () {}
    virtual bool apply (StudyReport const &rep, Sample *) const = 0;
    static Xtor *create (string const &name, Config const &conf);
};

#if 0
float xage (float age, float a, float b, float c) {
    float x = log(age) - log(25);
    return a * exp(- x * x / b) + c;
}
#endif

class ClinicalXtor: public Xtor {
    fs::path raw;
    int sample;
public:
    ClinicalXtor (Config const &conf):
        raw(conf.get<string>("adsb2.raw", "raw")),
        sample(conf.get<int>("adsb2.cli.sample", -1)) 
    {
    }

    bool apply (StudyReport const &rep,
                Sample *s) const {
        Study study(raw/fs::path(lexical_cast<string>(s->study))/fs::path("study"), false);
        vector<Slice *> slices;
        study.pool(&slices);
        if (slices.empty()) {
            LOG(ERROR) << "fail to load study " << s->study;
            return false;
        }
        if ((sample > 0) && (sample < slices.size())) {
            random_shuffle(slices.begin(), slices.end());
            slices.resize(sample);
        }
        Meta meta;
        cv::Mat mat;
        for (auto sl: slices) {
            mat = load_dicom(sl->path, &meta);
            if (mat.data) break;
            LOG(ERROR) << "fail to load DICOM: " << sl->path;
        }
        if (!mat.data) {
            LOG(ERROR) << "all DICOM paths failed for study " << s->study;
            return false;
        }
        s->tft_sys.clear();
        s->tft_sys.push_back(meta[Meta::SEX]);
        s->tft_sys.push_back(meta[Meta::AGE]);
        /*
        s->tft_sys.push_back(xage(meta[Meta::AGE], 181.712,6.073,1.821));
        s->tft_sys.push_back(xage(meta[Meta::AGE], 83.676,10.871,-6.782));
        */
        //s->tft_sys.push_back(meta[Meta::AGE]);
        s->tft_sys.push_back(meta[Meta::SLICE_THICKNESS]);
        s->tft_sys.push_back(meta.raw_spacing);
        s->tft_sys.push_back(meta.PercentPhaseFieldOfView);
        /*
        s->tft_sys.push_back(meta.width);
        s->tft_sys.push_back(meta.height);
        */
        s->tft_sys.push_back(meta.pos.x);
        s->tft_sys.push_back(meta.pos.y);
        /*
        s->tft_sys.push_back(meta.pos.z);
        */
        /*
        s->tft_sys.push_back(std::max(meta.AcquisitionMatrix[0], meta.AcquisitionMatrix[3]));
        */
        s->tft_dia = s->tft_sys;
        s->eft = s->tft_sys;
        s->cohort = meta.cohort;
        s->age = meta[Meta::AGE];
        return true;
    }
};

class OneSaxXtor: public Xtor {
    static void minmax (vector<SliceReport> const &sr, float *pm, float *pM) {
        float m = sr[0].data[SL_AREA];
        float M = m;
        for (unsigned i = 1; i < sr.size(); ++i){
            float v = sr[i].data[SL_AREA];
            if (v < m) v = m;
            if (v > M) v = M;
        }
        *pm = m;
        *pM = M;
    }
public:
    OneSaxXtor (Config const &conf) {
    }
    bool apply (StudyReport const &rep,
                Sample *s) const {
        if (rep.empty()) return false;
        auto meta = rep[0][0].meta;
#if 0
        float min = 0, max = 0;
        for (auto const &sr: rep) {
            float m, M;
            minmax(sr, &m, &M);
            if (M > max) {
                min = m;
                max = M;
            }
        }
#else
        float min, max;
        auto const &sr = rep[rep.size()/2];
        minmax(sr, &min, &max);
#endif

        s->tft_sys.clear();
        s->tft_sys.push_back(meta[Meta::SEX]);
        s->tft_sys.push_back(meta[Meta::AGE]);
        /*
        s->tft_sys.push_back(xage(meta[Meta::AGE], 181.712,6.073,1.821));
        s->tft_sys.push_back(xage(meta[Meta::AGE], 83.676,10.871,-6.782));
        */
        //s->tft_sys.push_back(meta[Meta::AGE]);
        s->tft_sys.push_back(meta[Meta::SLICE_THICKNESS]);
        s->tft_sys.push_back(meta.raw_spacing);
        s->tft_sys.push_back(meta.PercentPhaseFieldOfView);
        /*
        s->tft_sys.push_back(meta.width);
        s->tft_sys.push_back(meta.height);
        */
        s->tft_sys.push_back(meta.pos.x);
        s->tft_sys.push_back(meta.pos.y);
        s->tft_sys.push_back(min);
        s->tft_sys.push_back(max);
        /*
        s->tft.push_back(meta.pos.z);
        */
        /*
        s->tft.push_back(std::max(meta.AcquisitionMatrix[0], meta.AcquisitionMatrix[3]));
        */
        s->tft_dia = s->tft_sys;
        s->eft = s->tft_sys;
        s->cohort = rep[0][0].data[SL_COHORT];
        s->age = rep[0][0].meta[Meta::AGE];
        return true;
    }
};

bool do_xa = false;
class FullXtor: public Xtor {
    static bool compute1 (StudyReport const &rep, float *sys, float *dia) {
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

    static void compute2 (StudyReport const &rep, float *sys, float *dia) {
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

    static float compute_xa (StudyReport const &rep) {
        float vol = 0;
        for (unsigned i = 0; i + 1 < rep.size(); ++i) {
            float ss = 0;
            for (auto const &s: rep[i]) {
                ss += s.data[SL_XA];
            }
            ss /= rep[i].size();
            ss *= sqr(rep[i][0].meta.spacing);
            float gap = abs(rep[i][0].meta.slice_location - rep[i+1][0].meta.slice_location);
            if ((gap > 25)) gap = 10;
            vol += ss *gap;
        }
        return vol;
    }
public:
    FullXtor (Config const &conf) {
    }
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
        auto const &meta = rep[0][0].meta;
        vector<float> ft{
            meta[Meta::SEX], meta[Meta::AGE],
            //sys1, dia1, sys2, dia2//, compute_xa(rep)
            sys1, dia1, sys2,dia2
        };
        s->sys1 = sys1;
        s->dia1 = dia1;
        s->sys2 = sys2;
        s->dia2 = dia2;
        s->tft_sys = ft;
        s->tft_dia = ft;
        s->eft = ft;
        if (do_xa) {
            s->eft.push_back(compute_xa(rep));
        }
        s->cohort = rep[0][0].data[SL_COHORT];
        s->age = rep[0][0].meta[Meta::AGE];
        return true;
    }
};

Xtor *Xtor::create (string const &name, Config const &conf) {
    if (name == "cli") return new ClinicalXtor(conf);
    if (name == "one") return new OneSaxXtor(conf);
    if (name == "full") return new FullXtor(conf);
    CHECK(0);
    return nullptr;
}


#if 0
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
#endif
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
// train xgboost
void dump_ft (vector<float> const &ft, ostream &os) {
    for (unsigned i = 0; i < ft.size(); ++i) {
        os << " " << i << ":" << ft[i];
    }
}

void run_train (vector<Sample> &ss, int level, int mode, fs::path const &dir, unordered_set<int> const &train,
        fs::path const &model, int round, Config const &conf) {
    fs::create_directories(dir);
    CHECK(mode == 0 || mode == 1);
    CHECK(level == 1 || level == 2);
    fs::path train_path(dir/fs::path("train"));
    fs::path test_path(dir/fs::path("test"));

    fs::ofstream os_train(train_path);
    fs::ofstream os_test(test_path);
    int ntest = 0;
    for (auto &s: ss) {
        if (!s.good) continue;
        if (s.sys_t < 0) {
            LOG(ERROR) << "study " << s.study << " does not have groud truth sys but used for training";
            continue;
        }
        if (s.dia_t < 0) {
            LOG(ERROR) << "study " << s.study << " does not have groud truth dia but used for training";
            continue;
        }
        float target;
        vector<float> *ft;
        if (level == 1) {
            if (mode == 0) {
                target = s.sys_t;
                ft = &s.tft_sys;
            }
            else {
                target = s.dia_t;
                ft = &s.tft_dia;
            }
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
        if (round <= 0) {
            xg::TuneParams tp;
            xg::TuneResult tr;
            tp.max_it = conf.get<int>("adsb2.xg.max_it", 10);
            tp.max_round = conf.get<int>("adsb2.xg.max_round", 1500);
            tp.tolerate = conf.get<int>("adsb2.xg.tolerate", 0.5);
            tp.seed = conf.get<int>("adsb2.xg.seed", 2016);
            xg::tune(train_path, tp, &tr);
            round = (tr.round1 + tr.round2) / 2;
        }
        xg::Params params;
        params.round = round;
        xg::run_xgboost(train_path, ntest ? test_path : fs::path(), model,
                dir/fs::path("cmd"),
                dir/fs::path("cout"),
                dir/fs::path("cerr"),
                params);
    }
}

// v1 is true values
// v2 is predicted values
// evaluate RMSE & correlation
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

void run_eval (vector<Sample> &ss, unordered_set<int> const &train) {
    Eval eval;
    int count = 0;
    float sum = 0;
    float dsys = 0;
    float ddia = 0;
    float dall = 0;
    vector<float> sys_1, sys_2, sys_t;
    vector<float> dia_1, dia_2, dia_t;
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
        cout << "\t" << fmt::format("{:2.6f}", x);
    }
    /*
    cout << endl;
    cout << "sys2";
    */
    eval_v(sys_2, sys_t, &mm);
    for (auto x: mm) {
        cout << "\t" << fmt::format("{:2.6f}", x);
    }
    cout << endl;
    cout << "dia";
    eval_v(dia_1, dia_t, &mm);
    for (auto x: mm) {
        cout << "\t" << fmt::format("{:2.6f}", x);
    }
    /*
    cout << endl;
    cout << "dia2";
    */
    eval_v(dia_2, dia_t, &mm);
    for (auto x: mm) {
        cout << "\t" << fmt::format("{:2.6f}", x);
    }
    cout << endl;
    sys_t.insert(sys_t.end(), dia_t.begin(), dia_t.end());
    sys_1.insert(sys_1.end(), dia_1.begin(), dia_1.end());
    eval_v(sys_1, sys_t, &mm);
    cout << "all";
    for (auto x: mm) {
        cout << "\t" << fmt::format("{:2.6f}", x);
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

void run_pred (vector<Sample> &ss) {
    for (auto &s: ss) {
        cout << s.study << '\t' << s.sys_p << '\t' << s.sys_e << '\t' << s.dia_p << '\t' << s.dia_e
             << '\t' << s.sys_p_raw << '\t' << s.sys_e_raw << '\t' << s.dia_p_raw << '\t' << s.dia_e_raw << endl;
    }
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

#if 0
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
            (*data)[n].sys_v.swap(x);
        }
        else if (ss[1] == "Diastole") {
            (*data)[n].dia_v.swap(x);
        }
        else CHECK(0);
    }
}
#endif

void load_fallback (fs::path const &path, unordered_map<int, Fallback> *data) {
    data->clear();
    fs::ifstream is(path);
    string line;
    getline(is, line);  // first line
    while (getline(is, line)) {
        using namespace boost::algorithm;
        vector<string> ss;
        split(ss, line, is_any_of("\t"), token_compress_on);
        if (ss.size() != 9) {
            LOG(ERROR) << "Bad line: " << line;
        }
        int n = lexical_cast<int>(ss[0]);
        Fallback fb;
        fb.found = true;
        fb.sys_p = lexical_cast<float>(ss[1]);
        fb.sys_e = lexical_cast<float>(ss[2]);
        fb.dia_p = lexical_cast<float>(ss[3]);
        fb.dia_e = lexical_cast<float>(ss[4]);
        fb.sys_p_raw = lexical_cast<float>(ss[5]);
        fb.sys_e_raw = lexical_cast<float>(ss[6]);
        fb.dia_p_raw = lexical_cast<float>(ss[7]);
        fb.dia_e_raw = lexical_cast<float>(ss[8]);
        (*data)[n] = fb;
    }
}

void preprocess (StudyReport *rep, bool detail, bool smooth, Config const &conf) {
#if  0
    if (top) {
        patch_top_bottom(*rep);
    }
#endif
    //void Smooth (StudyReport *study, Config const &conf) {
    if (smooth) Smooth(rep, conf);
    if (detail) {
        for (auto &s: rep->back()) {
            s.data[SL_AREA] = 0;
        }
    }
}

static inline bool check_normal (float v) {
    return (v > -1000) && (v < 1000);
}

class FallbackChecker {
    float r1, r2, r3, r4;
public:
    FallbackChecker (Config const &conf)
        : r1(conf.get<float>("adsb2.fc.r1", 100)),
        r2(conf.get<float>("adsb2.fc.r2", 100)),
        r3(conf.get<float>("adsb2.fc.r3", 100)),
        r4(conf.get<float>("adsb2.fc.r4", 100))
    {

    }
    bool apply (Sample &s) {
        bool good = true;;
        if (!s.fb.found) return true;
        if (!check_normal(s.sys_p)) {
            LOG(ERROR) << "s.sys_p not normal: " << s.sys_p;
            s.sys_p = s.fb.sys_p;
            good = false;
        }
        if (!check_normal(s.dia_p)) {
            LOG(ERROR) << "s.dia_p not normal: " << s.dia_p;
            s.dia_p = s.fb.dia_p;
            good = false;
        }
        if (!check_normal(s.sys_e)) {
            LOG(ERROR) << "s.sys_e not normal: " << s.sys_e;
            s.sys_e = s.fb.sys_e;
            good = false;
        }
        if (!check_normal(s.dia_e)) {
            LOG(ERROR) << "s.dia_e not normal: " << s.dia_e;
            s.dia_e = s.fb.dia_e;
            good = false;
        }
        if (s.age < 10) return true;

        float sys_lb = s.fb.sys_p - s.fb.sys_e * r1;
        float sys_ub = s.fb.sys_p + s.fb.sys_e * r2;
        float dia_lb = s.fb.dia_p - s.fb.dia_e * r3;
        float dia_ub = s.fb.dia_p + s.fb.dia_e * r4;
        if ((s.sys_p < sys_lb) && (s.fb.sys_e > 0)) {
            float gain = numeric_limits<float>::quiet_NaN();
            if (s.sys_t > 0) {
                gain = fabs(s.sys_t - s.sys_p) - fabs(s.sys_t - sys_lb);
            }
            LOG(ERROR) << "study " << s.study << " sys " << s.sys_p << " => " << sys_lb << " age: " << s.age << " gain: " << gain << " type 1";
            s.sys_p = sys_lb;
            s.sys_e = s.fb.sys_e;
            good = false;
        }
        if ((s.fb.sys_p > 0) && (sys_ub > 0) && (s.sys_p > sys_ub)) {
            float gain = numeric_limits<float>::quiet_NaN();
            if (s.sys_t > 0) {
                gain = fabs(s.sys_t - s.sys_p) - fabs(s.sys_t - sys_ub);
            }
            LOG(ERROR) << "study " << s.study << " sys " << s.sys_p << " => " << sys_ub << " age: " << s.age << " gain: " << gain << " type 2";
            s.sys_p = sys_ub;
            s.sys_e = s.fb.sys_e;
            good = false;
        }
        if ((s.dia_p < dia_lb) && (s.fb.dia_e > 0)) {
            float gain = numeric_limits<float>::quiet_NaN();
            if (s.dia_t > 0) {
                gain = fabs(s.dia_t - s.dia_p) - fabs(s.dia_t - dia_lb);
            }
            LOG(ERROR) << "study " << s.study << " dia " << s.dia_p << " => " << dia_lb << " age: " << s.age << " gain: " << gain << " type 3";
            s.dia_p = dia_lb;
            s.dia_e = s.fb.dia_e;
            good = false;
        }
        if ((s.fb.dia_p > 0) && (dia_ub>0)&& (s.dia_p > dia_ub)) {
            float gain = numeric_limits<float>::quiet_NaN();
            if (s.dia_t > 0) {
                gain = fabs(s.dia_t - s.dia_p) - fabs(s.dia_t - dia_ub);
            }
            LOG(ERROR) << "study " << s.study << " dia " << s.dia_p << " => " << dia_ub << " age: " << s.age << " gain: " << gain << " type 4";
            s.dia_p = dia_ub;
            s.dia_e = s.fb.dia_e;
            good = false;
        }
        return good;
    }
};

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
    vector<fs::path> buddy_roots;
    fs::path fallback_path;
    fs::path fallback2_path;
    int round1, round2;
    float scale;
    int CASE;

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
    ("round1", po::value(&round1)->default_value(-1), "")
    ("round2", po::value(&round2)->default_value(-1), "")
    ("shuffle", "")
    ("data", po::value(&data_root), "dir containing report files")
    ("ws,w", po::value(&root), "working directory")
    ("fallback", po::value(&fallback_path), "")
    ("fallback2", po::value(&fallback2_path), "")
    ("xtor", po::value(&xtor_name)->default_value("full"), "")
    ("cohort", "")
    ("patch-cohort", "")
    ("buddy", po::value(&buddy_roots), "")
    ("xa", "")
    ("no-smooth", "")
    ("case", po::value(&CASE)->default_value(0), "")
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
    bool do_smooth = !(vm.count("no-smooth") > 0);
    bool do_cohort = vm.count("cohort") > 0;
    bool is_one = (xtor_name == "one");
    bool is_full = (xtor_name == "full");
    if (!fallback2_path.empty()) {
        CHECK(is_full);
        LOG(WARNING) << "Detected fallback2, make sure fallback2 is clinical model, fallback1 is one sax model.";
    }
    do_xa = vm.count("xa") > 0;

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
    unordered_map<int, Fallback> fallback;
    if (!fallback_path.empty()) {
        load_fallback(fallback_path, &fallback);
    }

    unordered_map<int, Fallback> fallback2;
    if (!fallback2_path.empty()) {
        load_fallback(fallback2_path, &fallback2);
    }

    // load cohort data
    unordered_map<int, int> patch_cohort;
    if (vm.count("patch-cohort")) {
        int id, c;
        fs::ifstream is(home_dir/fs::path("cohort"));
        while (is >> id >> c) {
            patch_cohort[id] = c;
        }
    }

    Eval eval;
    GaussianAcc gacc(config);
    Xtor *xtor = Xtor::create(xtor_name, config);
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
    else if (method == "pred") level = 2;
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

    FallbackChecker fbcheck(config);

    fs::create_directories(root);
    for (auto const &study: studies) {
        Sample s;
        s.study = study;
        s.good = true;
        s.cohort = 0;
        s.fb.found = false;
        s.fb2.found = false;
        // check and load fallback
        {
            auto it = fallback.find(s.study);
            if (it != fallback.end()) {
                s.fb = it->second;
            }
        }

        {
            auto it = fallback2.find(s.study);
            if (it != fallback2.end()) {
                s.fb2 = it->second;
            }
        }

        fs::path path(data_root / fs::path(lexical_cast<string>(study))/ fs::path("report.txt"));
        StudyReport x(path);
        if (x.empty()) {
            LOG(ERROR) << "Fail to load data file: " << path.native();
        }
#if 0
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
#endif
        preprocess(&x, do_detail, do_smooth, config);
        s.good = s.good && xtor->apply(x, &s);
        if (s.good && (!buddy_roots.empty())) {
            CHECK(xtor_name == "full");
            for (auto const &buddy_root: buddy_roots) {
                fs::path buddy_path = buddy_root / fs::path(lexical_cast<string>(s.study)) / fs::path("report.txt");
                StudyReport bx(buddy_path);
                if (bx.empty()) {
                    LOG(ERROR) << "Fail to load data file: " << buddy_path.native();
                }
                preprocess(&bx, do_detail, do_smooth, config);
                Sample bs;
                s.good = s.good && xtor->apply(bx, &bs);
                // 0 1 2 3     4   5
                //       dia1      dia2
                /*
                s.tft[3] = bs.tft[3];
                s.tft[5] = bs.tft[5];
                */
                /*
                s.tft.insert(s.tft.end(), bs.tft.begin() + 2, bs.tft.end());
                */
                s.eft.insert(s.eft.end(), bs.eft.begin() + 2, bs.eft.end());
                // s: sys   s.dia1 good, s.sys1 good
                // bs: dia  bs.dia1 good, bs.sys1 good

                // 0 1     2    3    4    5
                //        sys1  dia1 sys2 dia2
/*
model A: DIA.A1, DIA.A2, sys.a1, sys.a2
     
    bs   [3]     [5]     [4]     [2]
         
model B: dia.a1, dia.a2, SYS.A1, SYS.A2

     s   [3]     [5]     [4]     [2]

test1: DIA1, DIA2, SYS1, SYS2 ( you just did it today, right?, if not, test this first)

test 2: DIA.A1, SYS.A1.

test 3: for predicting DIA: DIA.A1, DIA.A2
for predicting SYS: SYS.A1, SYS.A2, DIA.A1, DIA.A2

test 4: for predicting DIA; DIA.A1.
for predicting SYS, SYS.A1, DIA.A1
*/
                if (CASE == 0) {
                    s.tft_sys.insert(s.tft_sys.end(), bs.tft_sys.begin() + 2, bs.tft_sys.end());
                    s.tft_dia.insert(s.tft_dia.end(), bs.tft_dia.begin() + 2, bs.tft_dia.end());
                }
                else if (CASE == 1) {
                    vector<float> tft{s.tft_sys[0], s.tft_sys[1],
                                bs.tft_sys[3], bs.tft_sys[5], s.tft_sys[4], s.tft_sys[2]};
                    s.tft_sys = tft;
                    s.tft_dia = tft;
                }
                else if (CASE == 2) {
                    vector<float> tft{s.tft_sys[0], s.tft_sys[1],
                                bs.tft_sys[3], s.tft_sys[4]};
                    s.tft_sys = tft;
                    s.tft_dia = tft;
                }
                else if (CASE == 3) {
                    vector<float> tft_sys{s.tft_sys[0], s.tft_sys[1],   // same as CASE 1 
                                bs.tft_sys[3], bs.tft_sys[5], s.tft_sys[4], s.tft_sys[2]};
                    vector<float> tft_dia{s.tft_sys[0], s.tft_sys[1],
                                bs.tft_sys[3], bs.tft_sys[5]};
                    s.tft_sys = tft_sys;
                    s.tft_dia = tft_dia;
                }
                else if (CASE == 4) {
                    vector<float> tft_sys{s.tft_sys[0], s.tft_sys[1],
                                s.tft_sys[4], bs.tft_sys[3]};
                    vector<float> tft_dia{s.tft_sys[0], s.tft_sys[1],
                                bs.tft_sys[3]};
                    s.tft_sys = tft_sys;
                    s.tft_dia = tft_dia;
                }
                else CHECK(0);
                
                

                
                break;
            }
        }

        s.sys_t = eval.get(s.study, 0);
        s.dia_t = eval.get(s.study, 1);
        int c_id = 0;
        if (do_cohort) {
            c_id = s.cohort;
            if (patch_cohort.size()) {
                auto it = patch_cohort.find(s.study);
                CHECK(it != patch_cohort.end());
                c_id = s.cohort = it->second;
            }
        }
        if (s.good) {
            if (level >= 1) {
                s.sys_p = target_sys[c_id]->apply(s.tft_sys);
                s.dia_p = target_dia[c_id]->apply(s.tft_dia);
                s.sys_p_raw = s.sys_p;
                s.dia_p_raw = s.dia_p;
            }
            if (level >= 2) {
                s.sys_e = error_sys->apply(s.eft) * scale;
                s.dia_e = error_dia->apply(s.eft) * scale;
                s.sys_e_raw = s.sys_e;
                s.dia_e_raw = s.dia_e;
            }
            // check fallback
            if (s.fb.found) {
                s.good &= fbcheck.apply(s);
            }
            if (is_one && s.fb.found) {
                if ((s.dia_p * 2 <= s.fb.dia_p_raw)
                     || (s.sys_e < 0) || (s.dia_e < 0) || (s.sys_p < 0)) {
                    LOG(WARNING) << "Study " << s.study << " (one) not good, using fallback";
                    s.sys_p = s.fb.sys_p;
                    s.sys_e = s.fb.sys_e;
                    s.dia_p = s.fb.dia_p;
                    s.dia_e = s.fb.dia_e;
                    s.good = false;
                }
            }
            if (is_full && s.fb.found && s.fb2.found) {
                float a = s.dia_p / (s.fb.dia_p_raw + s.fb2.dia_p_raw);
                float b = s.dia_p / s.fb.dia_p_raw;    // one sax
                float c = s.fb.dia_p / s.fb2.dia_p_raw;
                if ((b < 0.6) && (s.age > 10)) {
                    LOG(WARNING) << "Study " << s.study << "(full) not good, using fallback "
                                 << s.dia_p << " => " << s.fb.dia_p << " (" << s.dia_t << ")";
                    s.sys_p = s.fb.sys_p;
                    s.sys_e = s.fb.sys_e;
                    s.dia_p = s.fb.dia_p;
                    s.dia_e = s.fb.dia_e;
                    s.good = false;
                }
            }
        }
        else {
            LOG(WARNING) << "Study " << s.study << " not good, using fallback";
            if (s.fb.found) {
                s.sys_p = s.fb.sys_p;
                s.sys_e = s.fb.sys_e;
                s.dia_p = s.fb.dia_p;
                s.dia_e = s.fb.dia_e;
                s.sys_p_raw = s.fb.sys_p;
                s.sys_e_raw = s.fb.sys_e;
                s.dia_p_raw = s.fb.dia_p;
                s.dia_e_raw = s.fb.dia_e;
            }
            else {
                LOG(WARNING) << "Study " << s.study << " doesn not have fallback, using global value.";
                s.sys_p = global_fb.sys_p;
                s.sys_e = global_fb.sys_e;
                s.dia_p = global_fb.dia_p;
                s.dia_e = global_fb.dia_e;

                s.sys_p_raw = global_fb.sys_p;
                s.sys_e_raw = global_fb.sys_e;
                s.dia_p_raw = global_fb.dia_p;
                s.dia_e_raw = global_fb.dia_e;
            }
        }
        if (level >= 2) {
            gacc.apply(s.sys_p, s.sys_e, &s.sys_v);
            gacc.apply(s.dia_p, s.dia_e, &s.dia_v);
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
        run_train(c0, 1, 0, root/fs::path("d.target.sys.0"), train_set, root/fs::path("target.sys.0"), round1, config);
        run_train(c0, 1, 1, root/fs::path("d.target.dia.0"), train_set, root/fs::path("target.dia.0"), round1, config);
        if (do_cohort) {
            run_train(c1, 1, 0, root/fs::path("d.target.sys.1"), train_set, root/fs::path("target.sys.1"), round1, config);
            run_train(c1, 1, 1, root/fs::path("d.target.dia.1"), train_set, root/fs::path("target.dia.1"), round1, config);
        }
    }
    else if (method == "train2") {
        run_train(samples, 2, 0, root/fs::path("d.error.sys"), train_set, root/fs::path("error.sys"), round2, config);
        run_train(samples, 2, 1, root/fs::path("d.error.dia"), train_set, root/fs::path("error.dia"), round2, config);
    }
    else if (method == "eval") {
        run_eval(samples, train_set);
    }
    else if (method == "pred") {
        run_pred(samples);
    }
    else if (method == "submit") {
        run_submit(samples);
    }
    delete xtor;
    return 0;
}

void SmoothHelper (vector<SliceReport> *sax, 
        float mg, float Mg, float MM) {
    vector<float> a;
    for (auto &s: *sax) {
        a.push_back(s.data[SL_AREA]);
    }
    if (a.size() < 4) return;
    sort(a.begin(), a.end());
    a.pop_back();
    float smax = a.back();
    float max = std::min(smax + Mg, MM);
    //float min = a[1];
    float smin = a[1];
    float min = smin - mg;
    
    for (auto &s: *sax) {
        if (s.data[SL_AREA] > max) {
            s.data[SL_AREA] = max;
        }
        if (s.data[SL_AREA] < min) {
            s.data[SL_AREA] = min;
        }
    }
}

void Smooth (StudyReport *study, Config const &conf) {
    //float Mr = conf.get<float>("adsb2.smooth.Mr", 0);
    float Mg = conf.get<float>("adsb2.smooth.Mg", 20);
    float MM = conf.get<float>("adsb2.smooth.MM", 3200);
    //float mr = conf.get<float>("adsb2.smooth.mr", 0);
    float mg = conf.get<float>("adsb2.smooth.mg", 90);
#pragma omp parallel for
    for (unsigned i = 0; i < study->size(); ++i) {
        //SmoothHelper(&study->at(i), mr, mg, Mr, Mg);
        SmoothHelper(&study->at(i), mg, Mg, MM);
    }

}
