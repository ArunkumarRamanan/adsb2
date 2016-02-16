#define CPU_ONLY 1
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <boost/scoped_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>

#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>

#include "adsb2.h"

using namespace std;
using namespace boost;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace adsb2;

string backend("lmdb");
fs::path root;
fs::path sample_dir;
int sample_max = 100;
int sample_count = 0;

void FindBoundingCircle (cv::Mat label, cv::Point_<float> *C, float *R) {
    int min_x = label.cols;
    int max_x = -1;
    int min_y = label.rows;
    int max_y = -1;
    for (int y = 0; y < label.rows; ++y) {
        uint8_t const *row = label.ptr<uint8_t const>(y);
        for (int x = 0; x < label.cols; ++x) {
            if (row[x]) {   // positive pixel found
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }
    }
    CHECK(min_x <= max_x) << "no positive pixels are found.";
    *C = cv::Point_<float>(0.5 * (min_x + max_x), 0.5 * (min_y + max_y));
    int dx = max_x - min_x + 1;
    int dy = max_y - min_y + 1;
    *R = std::sqrt(dx * dx + dy * dy);
}

void import (Sampler &sampler,
             Cook &cook,
             vector<Slice *> &samples,
             fs::path const &dir,
             bool polar,
             int replica = 1) {

    CHECK(fs::create_directories(dir));
    CHECK(fs::is_directory(dir));
    fs::path image_path = dir / fs::path("images");
    fs::path label_path = dir / fs::path("labels");
  // Create new DB
    scoped_ptr<db::DB> image_db(db::GetDB(backend));
    image_db->Open(image_path.string(), db::NEW);
    scoped_ptr<db::Transaction> image_txn(image_db->NewTransaction());

    scoped_ptr<db::DB> label_db(db::GetDB(backend));
    label_db->Open(label_path.string(), db::NEW);
    scoped_ptr<db::Transaction> label_txn(label_db->NewTransaction());

    int count = 0;
    Slice tmp;
    for (unsigned rr = 0; rr < replica; ++rr) {
        if (rr) {
            random_shuffle(samples.begin(), samples.end());
        }
        cerr << "Loading replicate " << rr << "..." << endl;
        boost::progress_display progress(samples.size(), std::cerr);
#pragma omp parallel for
        for (unsigned id = 0; id < samples.size(); ++id) {
            Slice *dummy_sample = samples[id];
            Slice real_sample(*dummy_sample);
            real_sample.path = root / real_sample.path;
            real_sample.load_raw();
            CHECK(real_sample.images[IM_RAW].data);
            real_sample.path = dummy_sample->path;
            cook.apply(&real_sample);
            real_sample.anno->fill(real_sample, &real_sample.images[IM_LABEL], cv::Scalar(1));
            Slice *sample = &real_sample;

            CHECK(sample->images[IM_IMAGE].data);

            cv::Mat image, label;
            bool do_not_perturb = (rr == 0);
            if (polar) {
                cv::Point_<float> C;
                float R;
                FindBoundingCircle(sample->images[IM_LABEL], &C, &R);
                sampler.polar(sample->images[IM_IMAGE],
                              sample->images[IM_LABEL],
                              C, R,
                              &image, &label, do_not_perturb);
            }
            else {
                sampler.linear(sample->images[IM_IMAGE], sample->images[IM_LABEL],
                        &image, &label, do_not_perturb);

            }
            {
                CHECK(image.type() == CV_32F);
                cv::Mat u8;
                image.convertTo(u8, CV_8U);
                image = u8;
            }

            /*
            cv::rectangle(image, round(sample->box), cv::Scalar(0xFF));
            imwrite((boost::format("abc/%d.png") % count).str(), image);
            */
#pragma omp critical
            {
                Datum datum;
                string key = lexical_cast<string>(count), value;
                caffe::CVMatToDatum(image, &datum);
                datum.set_label(0);
                CHECK(datum.SerializeToString(&value));
                image_txn->Put(key, value);

                caffe::CVMatToDatum(label, &datum);
                datum.set_label(0);
                CHECK(datum.SerializeToString(&value));
                label_txn->Put(key, value);

                if (sample_count < sample_max) {
                    fs::path op(sample_dir / fs::path(fmt::format("s{}.jpg", sample_count++)));
                    Mat out;
                    vconcat(image, image + label * 255, out);
                    imwrite(op.native(), out);
                }

                if (++count % 1000 == 0) {
                    // Commit db
                    image_txn->Commit();
                    image_txn.reset(image_db->NewTransaction());
                    label_txn->Commit();
                    label_txn.reset(label_db->NewTransaction());
                }
                ++progress;
            }
        }
    }
    if (count % 1000 != 0) {
      image_txn->Commit();
      label_txn->Commit();
    }
}

void save_list (vector<Slice *> const &samples, fs::path path) {
    fs::ofstream os(path);
    for (auto const s: samples) {
        os << s->line << endl;
    }
}

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string list_path;
    string cbounds_path;
    string root_dir;
    string output_dir;
    string train_list_path;
    bool full = false;
    int F;
    int replica;
    bool polar = false;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("list", po::value(&list_path), "")
    ("cbounds", po::value(&cbounds_path)->default_value("/home/wdong/adsb2/data/color_bounds"), "color bound")
    ("root", po::value(&root_dir)->default_value("/ssd/wdong/adsb2/raw"), "")
    ("fold,f", po::value(&F)->default_value(1), "")
    ("full", "")
    ("train-list", po::value(&train_list_path), "using ids in this file for training, rest for validation")
    ("output,o", po::value(&output_dir), "")
    ("replica", po::value(&replica)->default_value(1), "")
    ("polar", "")
    ;

    po::positional_options_description p;
    p.add("list", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || list_path.empty() || output_dir.empty() || cbounds_path.empty()) {
        cerr << desc;
        return 1;
    }
    CHECK(F >= 1);
    full = vm.count("full") > 0;
    polar = vm.count("polar") != 0;
    root = fs::path(root_dir);

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);
    config.put("adsb2.cook.cbounds", cbounds_path);

    GlobalInit(argv[0], config);

    Cook cook(config);
    vector<Slice> samples;
    {
        ifstream is(list_path.c_str());
        string line;
        while (getline(is, line)) {
            samples.emplace_back(line);
        }
        cerr << samples.size() << " items loaded." << endl;
    }

    sample_dir = fs::path(output_dir) / fs::path("samples");
    fs::create_directories(sample_dir);
    CHECK(fs::is_directory(sample_dir));
    // generate labels
    Sampler sampler(config);

    if (train_list_path.size()) {
        unordered_set<int> pids;
        ifstream is(train_list_path.c_str());
        int i; 
        while (is >> i) {
            pids.insert(i);
        }
        vector<Slice *> train;
        vector<Slice *> val;
        for (auto &s: samples) {
            fs::path path = s.path;
            fs::path last;
            for (auto p: path) {
                if (p.native() == "study") break;
                last = p;
            }
            int n = lexical_cast<int>(last.native());
            if (pids.count(n)) {
                LOG(INFO) << "picked sample " << n << ": " << path;
                train.push_back(&s);
            }
            else {
                val.push_back(&s);
            }
            //ss[i] = &samples[i];
        }
        fs::path fold_path(output_dir);
        fs::create_directories(fold_path);
        CHECK(fs::is_directory(fold_path));
        cerr << "Loaded " << train.size() << " items for training." << endl;
        cerr << "Loaded " << val.size() << " items for validation." << endl;
        save_list(train, fold_path / fs::path("train.list"));
        save_list(val, fold_path / fs::path("val.list"));
        import(sampler, cook, train, fold_path / fs::path("train"), polar, replica);
        import(sampler, cook, val, fold_path / fs::path("val"), polar, 1);
        return 0;
    }

    if (F == 1) {
        vector<Slice *> ss(samples.size());
        for (unsigned i = 0; i < ss.size(); ++i) {
            ss[i] = &samples[i];
        }
        import(sampler, cook, ss, fs::path(output_dir), polar, replica);
        return 0;
    }
    // N-fold cross validation
    vector<vector<Slice *>> folds(F);
    random_shuffle(samples.begin(), samples.end());
    for (unsigned i = 0; i < samples.size(); ++i) {
        folds[i % F].push_back(&samples[i]);
    }

    for (unsigned f = 0; f < F; ++f) {
        vector<Slice *> &val = folds[f];
        // collect training examples
        vector<Slice *> train;
        for (unsigned i = 0; i < F; ++i) {
            if (i == f) continue;
            train.insert(train.end(), folds[i].begin(), folds[i].end());
        }
        fs::path fold_path(output_dir);
        if (full) {
            fold_path /= lexical_cast<string>(f);
        }
        fs::create_directories(fold_path);
        CHECK(fs::is_directory(fold_path));
        save_list(train, fold_path / fs::path("train.list"));
        save_list(val, fold_path / fs::path("val.list"));
        import(sampler, cook, train, fold_path / fs::path("train"), polar, replica);
        import(sampler, cook, val, fold_path / fs::path("val"), polar, 1);
        if (!full) break;
    }

    return 0;
}

