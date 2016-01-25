#define CPU_ONLY 1
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <boost/scoped_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
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

void import (ImageAugment const &aug,
             vector<Sample *> const &samples,
             fs::path const &dir, int channels) {
    CHECK(fs::create_directories(dir));
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
    for (Sample *sample: samples) {
        Datum datum;
        string key = lexical_cast<string>(sample->id), value;
        CHECK(sample->image.data);

        cv::Mat image, label;
        CaffeAdaptor::apply(*sample, &image, &label, channels);

        /*
        cv::rectangle(image, round(sample->box), cv::Scalar(0xFF));
        imwrite((boost::format("abc/%d.png") % count).str(), image);
        */

        caffe::CVMatToDatum(image, &datum);
        datum.set_label(0);
        CHECK(datum.SerializeToString(&value));
        image_txn->Put(key, value);

        caffe::CVMatToDatum(label, &datum);
        datum.set_label(0);
        CHECK(datum.SerializeToString(&value));
        label_txn->Put(key, value);
#if 0
        static int debug_count = 0;
        Mat out;
        rectangle(image, roi, Scalar(255));
        hconcat(image, label, out);
        imwrite((boost::format("%d.png") % debug_count).str(), out);
        ++debug_count;
#endif

        if (++count % 1000 == 0) {
            // Commit db
            image_txn->Commit();
            image_txn.reset(image_db->NewTransaction());
            label_txn->Commit();
            label_txn.reset(label_db->NewTransaction());
        }
    }
    if (count % 1000 != 0) {
      image_txn->Commit();
      label_txn->Commit();
    }
}

void save_list (vector<Sample *> const &samples, fs::path path) {
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
    ("fold,f", po::value(&F)->default_value(1), "")
    ("full", "")
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
    CHECK(F >= 1);
    full = vm.count("full") > 0;
    google::InitGoogleLogging(argv[0]);

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    Cook cook(config);
    int channels = config.get<int>("adsb2.caffe.channels", 1);
    ImageAugment aug(config);
    Samples samples(list_path, root_dir, cook);

    if (F == 1) {
        vector<Sample *> ss(samples.size());
        for (unsigned i = 0; i < ss.size(); ++i) {
            ss[i] = &samples[i];
        }
        import(aug, ss, fs::path(output_dir), channels);
        return 0;
    }
    // N-fold cross validation
    vector<vector<Sample *>> folds(F);
    random_shuffle(samples.begin(), samples.end());
    for (unsigned i = 0; i < samples.size(); ++i) {
        folds[i % F].push_back(&samples[i]);
    }

    for (unsigned f = 0; f < F; ++f) {
        vector<Sample *> const &val = folds[f];
        // collect training examples
        vector<Sample *> train;
        for (unsigned i = 0; i < F; ++i) {
            if (i == f) continue;
            train.insert(train.end(), folds[i].begin(), folds[i].end());
        }
        fs::path fold_path(output_dir);
        if (full) {
            fold_path /= lexical_cast<string>(f);
        }
        CHECK(fs::create_directories(fold_path));
        save_list(train, fold_path / fs::path("train.list"));
        save_list(val, fold_path / fs::path("val.list"));
        import(aug, train, fold_path / fs::path("train"), channels);
        import(aug, val, fold_path / fs::path("val"), channels);
        if (!full) break;
    }

    return 0;
}

