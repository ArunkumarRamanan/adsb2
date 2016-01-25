#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
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

    // OpenCV size and rect routines
    static inline cv::Rect round (cv::Rect_<float> const &f) {
        int x = std::round(f.x);
        int y = std::round(f.y);
        int width = std::round(f.x + f.width) - x;
        int height = std::round(f.y + f.height) - y;
        return cv::Rect(x, y, width, height);
    }

    template <typename T>
    static inline cv::Rect_<float>  operator * (cv::Rect_<T> &f, float scale) {
        return cv::Rect_<float>(f.x * scale,
                                f.y * scale,
                                f.width * scale,
                                f.height * scale);
    }

    static inline cv::Size round (cv::Size_<float> const &sz) {
        return cv::Size(std::round(sz.width), std::round(sz.height));
    }

    template <typename T>
    static inline cv::Size_<float> operator * (cv::Size_<T> const &sz, float scale) {
        return cv::Size_<float>(sz.width * scale, sz.height * scale);
    }

    struct Meta {
        float spacing;      // 
        float raw_spacing;  // original spacing as in file
        Meta (): spacing(-1), raw_spacing(-1) {
        }
    };

    cv::Mat load_dicom (fs::path const &, Meta *);

    struct Sample {
        int id;
        fs::path path;      // must always present
        Meta meta;
        cv::Mat raw;        // raw image, U16C1
        // size of the following 3 images must be the same
        cv::Mat image;      // cooked image             CV_32FC1
        cv::Mat vimage;   // cooked variance image    CV_32FC1
        //cv::Mat label;      // cooked label image,      CV_8UC1
        // the following lines might be missing
        string line;            // line as read from list file
        bool annotated;
        cv::Rect_<float> box;   // bouding box
        bool do_not_cook;

        Sample (): box(-1,-1,0,0), annotated(false), do_not_cook(false) {}
        Sample (string const &line);

        void load_raw () {
            raw = load_dicom(path, &meta);
#if 1   // Yuanfang's annotation assume images are all in landscape position
            if (raw.rows > raw.cols) {
                cv::transpose(raw, raw);
            }
#endif
            image = raw;
        }

        cv::Mat roi () {
            CHECK(box.x >=0 && box.y >= 0);
            cv::Rect roi = round(box);
            return image(roi);
        }

        void fill_roi (cv::Mat *mat, cv::Scalar const &v) const {
            CHECK(box.x >=0 && box.y >= 0);
            cv::Rect roi = round(box);
            (*mat)(roi).setTo(v);
        }

        void eval (cv::Mat mat, float *s1, float *s2) const;
    };

    struct ColorRange {
        int min, max;
        int umin, umax;
    };

    class Stack: public vector<Sample> {
    public:
        Stack ();
        // load from a directory of DCM files
        Stack (fs::path const &input_dir, bool load = true);

        cv::Size shape () const {
            return at(0).image.size();
        }

        void convert (int rtype, double alpha = 1, double beta = 0) {
            for (auto &s: *this) {
                s.image.convertTo(s.image, rtype, alpha, beta);
            }
        }

        void save_dir (fs::path const &dir, fs::path const &ext) {
            fs::create_directories(dir);
            for (auto const &s: *this) {
                fs::path path(dir / s.path.stem());
                path += ext;
                // fs::path((boost::format("%d.pgm") % i).str()));
                cv::imwrite(path.native(), s.image);
            }
        }

        void save_gif (string const &path) {
            fs::path tmp(fs::unique_path());
            fs::create_directories(tmp);
            ostringstream gif_cmd;
            gif_cmd << "convert -delay 5 ";
            fs::path ext(".pgm");
            for (auto const &s: *this) {
                fs::path pgm(tmp / s.path.stem());
                pgm += ext;
                cv::Mat u8;
                s.image.convertTo(u8, CV_8UC1);
                cv::imwrite(pgm.native(), u8);
                gif_cmd << " " << pgm;
            }
            gif_cmd << " " << fs::path(path);
            ::system(gif_cmd.str().c_str());
            fs::remove_all(tmp);
        }

        void getAvgStdDev (cv::Mat *avg, cv::Mat *stddev);
        void getColorRange (ColorRange *, float th = 0.9);
    };


    class Cook {
        float spacing;
        float color_vth;
        float color_eth;
        float color_margin;
        float color_max;
        int color_bins;
    public:
        Cook (Config const &config):
            spacing(config.get<float>("adsb2.cook.spacing", 1.0)),
            color_vth(config.get<float>("adsb2.color.vth", 0.99)),
            color_eth(config.get<float>("adsb2.color.eth", 0.1)),
            color_margin(config.get<float>("adsb2.color.margin", 0.04)),
            color_max(config.get<float>("adsb2.color.max", 255)),
            color_bins(config.get<float>("adsb2.cook.colors", 256))
        {
        }

        // cook the whole stack
        void apply (Stack *stack) const;
    };

    // samples do not belong to a single directory
    class Samples: public vector<Sample>
    {
    public:
        Samples (fs::path const &list_path, fs::path const &root, Cook const &cook);
    };


    class Detector {
    public:
        virtual ~Detector () {}
        virtual void apply (Sample &sample, cv::Mat *output) = 0;
    };


    Detector *make_caffe_detector (Config const &);
    Detector *make_cascade_detector (Config const &);
    Detector *make_scd_detector (Config const &);

    class ImageAugment {
    public:
        ImageAugment (Config const &config)
        {
        }
    };

    class CaffeAdaptor {
    public:
        static void apply (Sample &sample, cv::Mat *image, cv::Mat *label, int channels = 1) {
            CHECK(sample.image.type() == CV_32FC1);
            cv::Mat color;
            sample.image.convertTo(color, CV_8UC1);
            if (channels == 1) {
                *image = color;
            }
            else if (channels == 2) {
#if 0
                CHECK(sample.vimage.data);
                cv::Mat v;
                cv::normalize(sample.vimage, v, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::Mat z(v.size(), CV_8UC1, cv::Scalar(0));
                vector<cv::Mat> channels{color, v, z};
                cv::merge(channels, *image);
                CHECK(image->type() == CV_8UC3);
#else
                cv::normalize(sample.vimage, *image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
#endif
            }
            if (label) {
                CHECK(sample.annotated);
                label->create(color.size(), CV_8UC1);
                // save label
                label->setTo(cv::Scalar(0));
                sample.fill_roi(label, cv::Scalar(1));
            }
        }
    };

    template <typename I>
    void bound (I begin, I end, int *b, float margin) {
        float cc = 0;
        I it = begin;
        while (it < end) {
            cc += *it;
            if (cc >= margin) break;
            ++it;
        }
        *b = it - begin;
    }

    static inline void bound (vector<float> const &v, int *x, int *w, float margin) {
        int b, e;
        bound(v.begin(), v.end(), &b, margin);
        bound(v.rbegin(), v.rend(), &e, margin);
        e = v.size() - e;
        if (e < b) e = b;
        *x = b;
        *w = e - b;
    }

    static inline void bound (cv::Mat const &image, cv::Rect *rect, float th) {
        vector<float> X(image.cols, 0);
        vector<float> Y(image.rows, 0);
        float total = 0;
        CHECK(image.type() == CV_32F);
        for (int y = 0; y < image.rows; ++y) {
            float const *row = image.ptr<float>(y);
            for (int x = 0; x < image.cols; ++x) {
                float v = row[x];
                X[x] += v;
                Y[y] += v;
                total += v;
            }
        }
        float margin = total * (1.0 - th) / 2;
        bound(X, &rect->x, &rect->width, margin);
        bound(Y, &rect->y, &rect->height, margin);
    }

    template <typename T>
    void percentile (vector<T> &all, vector<float> const &p, vector<T> *v) {
        sort(all.begin(), all.end());
        v->resize(p.size());
        for (unsigned i = 0; i < p.size(); ++i) {
            int t = int(std::floor(all.size() * p[i]));
            if (t < 0) t = 0;
            if (t >= all.size()) t = all.size() - 1;
            v->at(i) = all[t];
        }
    }

    template <typename T>
    void percentile (cv::Mat const &mat, vector<float> const &p, vector<T> *v) {
        vector<T> all;
        CHECK(mat.total());
        for (int i = 0; i < mat.rows; ++i) {
            T const *p = mat.ptr<T const>(i);
            for (int j = 0; j < mat.cols; ++j) {
                all.push_back(p[j]);
            }
        }
        percentile(all, p, v);
    }

    template <typename T>
    T percentile (cv::Mat const &mat, float p) {
        vector<T> v;
        percentile(mat, vector<float>{p}, &v);
        return v[0];
    }
}
