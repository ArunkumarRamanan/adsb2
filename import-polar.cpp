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
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>
#include <json11.hpp>

#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>

#include "adsb2.h"

using namespace std;
using namespace boost;
using namespace cv;
using namespace json11;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace adsb2;

string backend("lmdb");

struct Sample {
    Slice slice;
    cv::Mat label;
    cv::Point_<float> c;
    float R;
    vector<cv::Point_<float>> cc;

    void load (float spacing) {
        slice.load_raw();
        slice.raw.convertTo(slice.image, CV_32F);
        if (spacing > 0) {
            float ratio = slice.meta.raw_spacing / spacing;
            cv::Size sz = round(slice.raw.size() * ratio);
            c.x *= ratio;
            c.y *= ratio;
            R *= ratio;
            slice.meta.spacing = spacing;
            cv::resize(slice.image, slice.image, sz);
        }
        //
        //  0 -- front           back --bottom
        //  back -- | -- fromt
        int xx = 0;
        {
            float bx = cc.back().x;
            float by = 1 - cc.back().y;
            float fx = cc.front().x;
            float fy = cc.front().y;
            xx = std::round((by * fx + fy * bx) / (fy + by) * slice.image.cols);
        }
        vector<cv::Point> ps(cc.size());
        for (unsigned i = 0; i < cc.size(); ++i) {
            auto const &from = cc[i];
            auto &to = ps[i];
            to.x = std::round(slice.image.cols * from.x);
            to.y = std::round(slice.image.rows * from.y);
        }
        ps.emplace_back(xx, slice.image.rows - 1);
        ps.emplace_back(0, slice.image.rows - 1);
        ps.emplace_back(0, 0);
        ps.emplace_back(xx, 0);
        Point const *pps = &ps[0];
        int const nps = ps.size();
        cv::Mat polar(slice.image.size(), CV_32F, cv::Scalar(0));
        cv::fillPoly(polar, &pps, &nps, 1, cv::Scalar(1));
        //cv::polylines(polar, &pps, &nps, 1, true, cv::Scalar(255), 2);
        //imwrite("/home/wdong/public_html/xxx.png", polar);
        linearPolar(polar, &label, c, R, CV_INTER_NN+CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);
        /*
        cv::Mat out;
        slice.image.convertTo(out, CV_8U);
        out += label;
        cv::Mat oo = slice.image + label;
        cv::hconcat(oo, label, slice.image);
        imwrite("/home/wdong/public_html/yyy.png", slice.image);
        */
    }
};

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    string list_path;
    string root_dir;
    string output_dir;
    int round;
    bool do_polar = false;
    float min_R;
    float max_R;
    float max_C;
    int mk;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("list", po::value(&list_path), "")
    ("root", po::value(&root_dir), "")
    ("output,o", po::value(&output_dir), "")
    ("aug", po::value(&round)->default_value(1), "")
    ("polar", "")
    ("min_R", po::value(&min_R)->default_value(0.75), "")
    ("max_R", po::value(&max_R)->default_value(1.5), "")
    ("max_C", po::value(&max_C)->default_value(0.4), "")
    ("mk", po::value(&mk)->default_value(3), "")
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

    if (vm.count("polar")) do_polar = true;


    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);

    Cook cook(config);
    int channels = config.get<int>("adsb2.caffe.channels", 1);
    ImageAugment aug(config);

    vector<Sample> samples;
    {
        LOG(INFO) << "Loading samples.";
        ifstream is(list_path.c_str());
        string line;
        fs::path root(root_dir);
        while (getline(is, line)) {
            Json dcm_meta, con_meta;
            {
                using namespace boost::algorithm;
                vector<string> ss;
                split(ss, line, is_any_of("\t"), token_compress_on);
                CHECK(ss.size() == 3);
                string err;
                dcm_meta = Json::parse(ss[1], err);
                CHECK(err.empty());
                con_meta = Json::parse(ss[2], err);
                CHECK(err.empty());
            }
            Sample s;
            s.slice.path = root / fs::path(dcm_meta["dcm"].string_value());
            s.c.x = dcm_meta["x"].number_value();
            s.c.y = dcm_meta["y"].number_value();
            s.R = dcm_meta["R"].number_value();
            for (auto const &pp: con_meta["shapes"][0]["geometry"]["points"].array_items()) {
                float rho = pp["x"].number_value();
                float phi = pp["y"].number_value();
                if (rho < 0.5) continue;
                rho -= 0.5;
                rho *= 2;
                s.cc.emplace_back(rho, phi);
            }
            sort(s.cc.begin(), s.cc.end(), [](cv::Point_<float> const &p1,
                                              cv::Point_<float> const &p2) {
                        return p1.y < p2.y;
                    });
            samples.push_back(s);
        }
    }
    float spacing(config.get<float>("adsb2.cook.spacing", 1.4));
    for (auto &s: samples) {
        s.load(spacing);
    }
    vector<Sample *> ss;
    for (auto &s: samples) {
        ss.push_back(&s);
    }
    fs::path dir(output_dir);
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
    Slice tmp;
    uniform_real_distribution<float> R_R(min_R, max_R);
    uniform_real_distribution<float> C_R(0, max_C);
    uniform_real_distribution<float> C_RHO(0, M_PI * 2);
    default_random_engine rng;
    cv::Mat kernel = cv::Mat::ones(mk, mk, CV_8U);
    for (unsigned rr = 0; rr < round; ++rr) {
        random_shuffle(ss.begin(), ss.end());
        for (Sample *sample: ss) {

            Datum datum;
            string key = lexical_cast<string>(count), value;
            CHECK(sample->slice.image.data);

            cv::Mat image, label;
            if (do_polar) {
                float rr = R_R(rng) * sample->R;
                float cr = C_R(rng) * sample->R;
                float rho = C_RHO(rng);
                cv::Point_<float> cc = sample->c + cv::Point_<float>(cr * cos(rho), cr * sin(rho));
                cv::Mat imageF, labelF;
                linearPolar(sample->slice.image, &imageF, cc, rr, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
                linearPolar(sample->label, &labelF, cc, rr, CV_INTER_NN+CV_WARP_FILL_OUTLIERS);
                imageF.convertTo(image, CV_8UC1);
                //cv::equalizeHist(image, image);
                labelF.convertTo(label, CV_8UC1);
                cv::morphologyEx(label, label, cv::MORPH_CLOSE, kernel);
                /*
                cv::Mat tmp;
                cv::hconcat(image, label * 255, tmp);
                fs::path p(dir);
                p /= fs::path(fmt::format("{}.png", count));
                cv::imwrite(p.native(), tmp);
                */
            }
            else {
                CHECK(0);
#if 0
                if (rr) {
                    CHECK(channels == 1);
                    aug.apply(*sample, &tmp);
                    CaffeAdaptor::apply(tmp, &image, &label, channels, do_circle);
                }
                else {
                    CaffeAdaptor::apply(*sample, &image, &label, channels, do_circle);
                }
#endif
            }

            caffe::CVMatToDatum(image, &datum);
            datum.set_label(0);
            CHECK(datum.SerializeToString(&value));
            image_txn->Put(key, value);

            caffe::CVMatToDatum(label, &datum);
            datum.set_label(0);
            CHECK(datum.SerializeToString(&value));
            label_txn->Put(key, value);

            if (++count % 1000 == 0) {
                // Commit db
                image_txn->Commit();
                image_txn.reset(image_db->NewTransaction());
                label_txn->Commit();
                label_txn.reset(label_db->NewTransaction());
            }
        }
    }
    if (count % 1000 != 0) {
      image_txn->Commit();
      label_txn->Commit();
    }

    return 0;
}

