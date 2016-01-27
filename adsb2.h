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
        struct Study {
            string part;        // HEART
            char sex;
            float age;
            bool operator == (Study const &s) const {
                return (part == s.part)
                    && (sex == s.sex)
                    && (age == s.age);
            }
        };
        struct Series {
            float slice_thickness;  // mm
            //float slice_spacing;    // mm
            float nominal_interval;
            float repetition_time;
            float echo_time;
            int number_of_images;
            float slice_location;
            int series_number;
            bool operator == (Series const &s) const {
                return (slice_thickness == s.slice_thickness)
                    && (nominal_interval == s.nominal_interval)
                    && (repetition_time == s.repetition_time)
                    && (echo_time == s.echo_time)
                    && (number_of_images == s.number_of_images)
                    && (slice_location == s.slice_location)
                    && (series_number == s.series_number);
            }
        };
        Study study;
        Series series;
        float trigger_time;
        //float repetition_time;
        //float echo_time;
        float spacing;      //  mm
        float raw_spacing;  // original spacing as in file
        Meta (): spacing(-1), raw_spacing(-1) {
        }
    };

    void GlobalInit (char const *path, Config const &config);
    void dicom_setup (char const *path, Config const &config);
    cv::Mat load_dicom (fs::path const &, Meta *);

    struct Slice {
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

        cv::Mat prob;

        Slice (): box(-1,-1,0,0), annotated(false), do_not_cook(false) {}
        Slice (string const &line);

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

    class Series: public vector<Slice> {
        fs::path series_path;
        void sanity_check () {
            CHECK(size());
            CHECK(at(0).meta.series.number_of_images == size());
            for (unsigned i = 1; i < size(); ++i) {
                CHECK(at(i).meta.study == at(0).meta.study);
                CHECK(at(i).meta.series == at(0).meta.series);
                CHECK(at(i).meta.trigger_time
                            > at(i-1).meta.trigger_time);
            }
        }
    public:
        Series (){}
        // load from a directory of DCM files
        Series (fs::path const &input_dir, bool load = true);

        fs::path dir () const {
            return series_path;
        }

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

    static constexpr float LOCATION_GAP_EPSILON = 0.01;
    static inline bool operator < (Series const &s1, Series const &s2) {
        Meta::Series const &m1 = s1.front().meta.series;
        Meta::Series const &m2 = s2.front().meta.series;
        if (m1.slice_location + LOCATION_GAP_EPSILON < m2.slice_location) {
            return true;
        }
        if (m1.slice_location - LOCATION_GAP_EPSILON > m2.slice_location) {
            return false;
        }
        return m1.series_number < m2.series_number;
    }

    class Study: public vector<Series> {
        fs::path study_path;
        void fix_order () {
            sort(begin(), end());
            unsigned off = 1;
            for (unsigned i = 1; i < size(); ++i) {
                Meta::Series const &prev = at(off-1).front().meta.series;
                Meta::Series const &cur = at(i).front().meta.series;
                if (std::abs(prev.slice_location - cur.slice_location) <= LOCATION_GAP_EPSILON) {
                    LOG(WARNING) << "replacing " << at(off-1).dir()
                                 << " (" << prev.slice_location << ") "
                                 << " with " << at(i).dir()
                                 << " (" << cur.slice_location << ") ";
                    std::swap(at(off-1), at(i));
                }
                else {
                    if (off != i) { // otherwise no need to swap
                        std::swap(at(off), at(i));
                    }
                    ++off;
                }
            }
            if (off != size()) {
                LOG(WARNING) << "study " << study_path << " reduced from " << size() << " to " << off << " series.";
            }
            resize(off);
        }
        void sanity_check () {
            CHECK(size());
            for (unsigned i = 1; i < size(); ++i) {
                CHECK(at(i).front().meta.study == at(0).front().meta.study);
                CHECK(at(i).front().meta.series.slice_thickness 
                        == at(i-1).front().meta.series.slice_thickness);
                CHECK(at(i).front().meta.series.number_of_images
                        == at(i-1).front().meta.series.number_of_images);
                CHECK(at(i).front().meta.series.slice_location
                        > at(i-1).front().meta.series.slice_location);
            }
            /*
            CHECK(at(0).meta.series.number_of_images == size());
            for (unsigned i = 1; i < size(); ++i) {
                CHECK(at(i).meta.study == at(0).meta.study);
                CHECK(at(i).meta.series == at(0).meta.series);
                CHECK(at(i).meta.trigger_time
                            > at(i-1).meta.trigger_time);
            }
            */
        }
    public:
        Study ();
        // load from a directory of DCM files
        Study (fs::path const &input_dir, bool load = true);
    };


    class Cook {
        float spacing;
        int color_bins;
    public:
        Cook (Config const &config):
            spacing(config.get<float>("adsb2.cook.spacing", 1.4)),
            color_bins(config.get<float>("adsb2.cook.colors", 256))
        {
        }

        // cook the whole stack
        void apply (Series *stack) const;
    };

    // samples do not belong to a single directory
    class Slices: public vector<Slice>
    {
    public:
        Slices (fs::path const &list_path, fs::path const &root, Cook const &cook);
    };


    class Detector {
    public:
        virtual ~Detector () {}
        virtual void apply (Slice *sample) = 0;
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
        static void apply (Slice &sample, cv::Mat *image, cv::Mat *label, int channels = 1) {
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

    class Gaussian {
        cv::Mat mean;
        cv::Mat cov;
        cv::Mat icov;
    public:
        Gaussian (cv::Mat samples, cv::Mat weights)
            : mean(1, samples.cols, CV_32F, cv::Scalar(0)),
            cov(samples.cols, samples.cols, CV_32F, cv::Scalar(0))
        {
            CHECK(samples.rows == weights.rows);
            CHECK(weights.cols == 1);
            float sum = 0;
            for (int i = 0; i < samples.rows; ++i) {
                float w = weights.ptr<float>(i)[0];
                mean += w * samples.row(i);
                sum += w;
            }
            mean /= sum;
            for (int i = 0; i < samples.rows; ++i) {
                cv::Mat row = samples.row(i) - mean;
                cov += weights.ptr<float>(i)[0] * row.t() * row;
            }
            cov /= sum;
            icov = cov.inv(cv::DECOMP_SVD);
            LOG(INFO) << "mean: " << mean;
            LOG(INFO) << "cov: " << cov;
            LOG(INFO) << "icov: " << icov;
        }
        cv::Mat prob (cv::Mat in) const {
            CHECK(in.cols == mean.cols);
            cv::Mat r(in.rows, 1, CV_32F);
            float *ptr = r.ptr<float>(0);
            for (int i = 0; i < in.rows; ++i) {
                cv::Mat r = in.row(i) - mean;
                cv::Mat k = r * icov * r.t();
                CHECK(k.rows == 1);
                CHECK(k.cols == 1);
                float v = std::exp(-0.5 * k.ptr<float>(0)[0]);
                ptr[i] = v;
            }
            return r;
        }
    };

    void Var2Prob (cv::Mat in, cv::Mat *out, float pth, int mk);
}
