#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/ptree.hpp>
#include <boost/assert.hpp>
#include <glog/logging.h>

namespace adsb2 {

    using std::string;
    using std::vector;
    using std::ostringstream;
    using std::istringstream;
    using std::ifstream;
    namespace fs = boost::filesystem;

    // XML configuration
    typedef boost::property_tree::ptree Config;

    void LoadConfig (string const &path, Config *);
    void SaveConfig (string const &path, Config const &);
    // Overriding configuration options in the form of "KEY=VALUE"
    void OverrideConfig (std::vector<std::string> const &overrides, Config *);

    class Detector {
    public:
        virtual ~Detector () {}
        virtual void apply (cv::Mat input, cv::Mat *output) = 0;
    };

    Detector *make_caffe_detector (string const &);
    Detector *make_cascade_detector (string const &);

    struct Meta {
        float pixel_spacing;
    };

    class ImageLoader {
    public:
        ImageLoader (Config const &config) {
        }

        cv::Mat load (string const &path, Meta *meta = nullptr) const;
    };

    class Stack: public vector<cv::Mat> {
    public:
        cv::Size shape () const {
            return at(0).size();
        }

        void convert (int rtype, double alpha = 1, double beta = 0) {
            for (auto &image: *this) {
                image.convertTo(image, rtype, alpha, beta);
            }
        }

        void make_gif (string const &path) {
            fs::path tmp(fs::unique_path());
            fs::create_directories(tmp);
            ostringstream gif_cmd;
            gif_cmd << "convert -delay 5 ";
            for (unsigned i = 0; i < size(); ++i) {
                fs::path pgm(tmp / fs::path((boost::format("%d.pgm") % i).str()));
                cv::imwrite(pgm.native(), at(i));
                gif_cmd << " " << pgm;
            }
            gif_cmd << " " << fs::path(path);
            ::system(gif_cmd.str().c_str());
            fs::remove_all(tmp);
        }
    };

    class DcmStack: public Stack {
        vector<fs::path> names;
    public:
        DcmStack (std::string const &dir, ImageLoader const &loader) {
            // enumerate DCM files
            fs::path input_dir(dir);
            fs::directory_iterator end_itr;
            for (fs::directory_iterator itr(input_dir);
                    itr != end_itr; ++itr) {
                if (fs::is_regular_file(itr->status())) {
                    // found subdirectory,
                    // create tagger
                    auto path = itr->path();
                    auto stem = path.stem();
                    auto ext = path.extension();
                    if (ext.string() != ".dcm") {
                        LOG(WARNING) << "Unknown file type: " << path.string();
                        continue;
                    }
                    names.push_back(stem);
                }
            }
            std::sort(names.begin(), names.end());
            resize(names.size());
            for (unsigned i = 0; i < names.size(); ++i) {
                auto const &name = names[i];
                auto dcm_path = input_dir;
                dcm_path /= name;
                dcm_path += ".dcm";
                cv::Mat image = loader.load(dcm_path.native());
                BOOST_VERIFY(image.total());
                BOOST_VERIFY(image.type() == CV_8UC1);
                BOOST_VERIFY(image.isContinuous());
                if (i) {
                    BOOST_VERIFY(image.size() == at(0).size());
                }
                at(i) = image;
            }
        }
    };

    static inline void round (cv::Rect_<float> const &from, cv::Rect *to) {
        to->x = std::round(from.x);
        to->y = std::round(from.y);
        to->width = std::round(from.x + from.width) - to->x;
        to->height = std::round(from.y + from.height) - to->y;
    }


    struct Sample {
        int id;
        string line;
        string path;
        cv::Rect_<float> box;

        bool load (string const &txt) {
            istringstream ss(txt);
            ss >> path >> box.x >> box.y >> box.width >> box.height;
            if (!ss) return false;
            line = txt;
            return true;
        }

        void fill_roi (cv::Mat *mat, cv::Scalar const &v) const {
            cv::Rect roi;
            round(box, &roi);
            (*mat)(roi).setTo(v);
        }

        void eval (cv::Mat mat, float *s1, float *s2) const;
    };

    class Samples: public vector<Sample> {
    public:
        Samples (string const &path, string const &root_dir) {
            ifstream is(path.c_str());
            CHECK(is) << "Cannot open list file: " << path;
            Sample s;
            s.id = 0;
            string line;
            while (getline(is, line)) {
                if (!s.load(line)) {
                    LOG(ERROR) << "Bad line: " << line;
                    continue;
                }
                if (!fs::is_regular_file(fs::path(root_dir + s.path))) {
                    LOG(ERROR) << "Cannot find regular file: " << s.path;
                    continue;
                }
                push_back(s);
                ++s.id;
            }
            LOG(INFO) << "Loaded " << size() << " samples.";
        }
    };

}
