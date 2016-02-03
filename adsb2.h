#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <opencv2/opencv.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/ptree.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/timer/timer.hpp>
#include <boost/assert.hpp>
#include <cppformat/format.h>
#include <glog/logging.h>

namespace adsb2 {

    using std::string;
    using std::vector;
    using std::ostringstream;
    using std::istringstream;
    using std::ifstream;
    using boost::lexical_cast;
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

    static inline cv::Rect_<float> unround (cv::Rect const &r) {
        return cv::Rect_<float>(r.x, r.y, r.width, r.height);
    }

    static inline cv::Rect_<float> cscale (cv::Rect_<float> const &r, float scale) {
        float cx = r.x + r.width/2;
        float cy = r.y + r.height/2;
        float w = r.width * scale;
        float h = r.height * scale;
        return cv::Rect_<float>(cx - w/2, cy - h/2, w, h);
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

    static inline cv::Point round (cv::Point_<float> const &pt) {
        return cv::Point(std::round(pt.x), std::round(pt.y));
    }

    template <typename T>
    static inline cv::Size_<float> operator * (cv::Size_<T> const &sz, float scale) {
        return cv::Size_<float>(sz.width * scale, sz.height * scale);
    }

    struct MetaBase {
        enum {
            SEX = 0,    // male: 0, female: 1
            AGE,
            SLICE_THICKNESS,
            NOMINAL_INTERVAL,
            NUMBER_OF_IMAGES,
            SLICE_LOCATION,
            SERIES_NUMBER,

            SERIES_FIELDS,
            STUDY_FIELDS = 2,   // the first 2 fields are study fields
        };
        float trigger_time;
        float spacing;      //  mm
        float raw_spacing;  // original spacing as in file
        MetaBase (): spacing(-1), raw_spacing(-1) {
        }
        static char const *FIELDS[];
    };

    struct Meta: public MetaBase, public std::array<float, MetaBase::SERIES_FIELDS> {
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
        cv::Rect pred;  // prediction
        bool do_not_cook;

        cv::Mat prob;

        Slice (): box(-1,-1,0,0), annotated(false), do_not_cook(false) {}
        Slice (string const &line);

        void clone (Slice *s) {
            s->id = id;
            s->path = path;
            s->meta = meta;
            s->raw = raw.clone();
            s->image = image.clone();
            s->vimage = vimage.clone();
            s->line = line;
            s->annotated = annotated;
            s->box = box;
            s->pred = pred;
            s->do_not_cook = do_not_cook;
            s->prob = prob.clone();
        }

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

        void fill_circle (cv::Mat *mat, cv::Scalar const &v) const {
            cv::circle(*mat, round(cv::Point_<float>(box.x + box.width/2, box.y + box.height/2)),
                    std::sqrt(box.width * box.height)/2, v, -1);
        }

        void eval (cv::Mat mat, float *s1, float *s2) const;
    };

    struct ColorRange {
        int min, max;
        int umin, umax;
    };

    class Series: public vector<Slice> {
        fs::path series_path;
        bool sanity_check (bool fix = false);
        friend class Study;
    public:
        Series (){}
        // load from a directory of DCM files
        Series (fs::path const &input_dir, bool load = true, bool check = true, bool fix = false);

        void bound (cv::Rect bb) {
            bb = bb & cv::Rect(cv::Point(0,0), front().image.size());
            cv::Mat vimage = front().vimage(bb).clone();
            for (Slice &s: *this) {
                s.image = s.image(bb).clone();
                s.vimage = vimage;
                CHECK(!s.prob.data);
                if (s.annotated) {
                    s.box -= cv::Point_<float>(bb.x, bb.y);
                }
            }
        }

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

        void save_gif (fs::path const &path) {//string const &path) {
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
            gif_cmd << " " << path;
            ::system(gif_cmd.str().c_str());
            fs::remove_all(tmp);
        }

        void save_gif (string const &path) {
            save_gif(fs::path(path));
        }

        void getAvgStdDev (cv::Mat *avg, cv::Mat *stddev);
        void getColorRange (ColorRange *, float th = 0.9);
        void visualize (bool show_prob = true);
    };

    class Study: public vector<Series> {
        fs::path study_path;
        bool sanity_check (bool fix = false);
        void check_regroup ();  // some times its necessary to regroup one series into
                                // multiple series
    public:
        Study ();
        // load from a directory of DCM files
        Study (fs::path const &input_dir, bool load = true, bool check = true, bool fix = false);
        fs::path dir () const {
            return study_path;
        }
        void bound (cv::Rect const &bb) {
            for (auto &s: *this) {
                s.bound(bb);
            }
        }
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
        void apply (Study *stucy) const;
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
        virtual void apply (cv::Mat image, cv::Mat *prob) = 0;
    };


    Detector *make_caffe_detector (Config const &);
    Detector *make_caffe_detector (string const &path);
    Detector *make_cascade_detector (Config const &);
    Detector *make_scd_detector (Config const &);

    class ImageAugment {
        float max_color;
        float max_angle;
        float max_scale;
        std::default_random_engine e;
        float sample (float limit) {
            std::uniform_real_distribution<float> dis(-limit, limit);
            return dis(e);
        }
    public:
        ImageAugment (Config const &config)
            : max_color(config.get<float>("adsb2.aug.color", 20)),
            max_angle(config.get<float>("adsb2.aug.angle", 10.0 * M_PI / 180)),
            max_scale(config.get<float>("adsb2.aug.scale", 1.2))
        {
        }

        void apply (Slice &from, Slice *to) {
            CHECK(from.annotated);
            float color(sample(max_color));
            float angle(sample(max_angle));
            float scale(std::exp(sample(std::log(max_scale))));
            bool flip((e() % 2) == 1);
            vector<float> cc{from.box.x + 1.0f*from.box.width/2,
                             from.box.y + 1.0f*from.box.height/2};
            cv::Mat image;
            if (flip) {
                cv::flip(from.image, image, 1);
                cc[0] = image.cols - cc[0];
            }
            else {
                image = from.image;
            }
            cv::Mat rot = cv::getRotationMatrix2D(cv::Point(image.cols/2, image.rows/2), angle, scale);
            cv::warpAffine(image, to->image, rot, image.size());
            to->image += color;
            {
                cv::Mat cm(1, 1, CV_32FC2, &cc[0]);
                cv::transform(cm, cm, rot);
            }
            float r = from.box.width * scale;
            to->annotated = true;
            to->box.x = std::round(cc[0] - r/2);
            to->box.y = std::round(cc[1] - r/2);
            to->box.width = to->box.height = std::round(r);
        }
    };

    class CaffeAdaptor {
    public:
        static void apply (Slice &sample, cv::Mat *image, cv::Mat *label, int channels = 1, bool circle = false) {
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
                if (circle) {
                    sample.fill_circle(label, cv::Scalar(1));
                }
                else {
                    sample.fill_roi(label, cv::Scalar(1));
                }
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

    float accumulate (cv::Mat const &image, vector<float> *X, vector<float> *Y);

    static inline void bound (cv::Mat const &image, cv::Rect *rect, float th) {
        vector<float> X;
        vector<float> Y;
        float total = accumulate(image, &X, &Y);
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

    void Bound (Detector *det, Study *study, cv::Rect *box, Config const &config);
    // use variance image (big variance == big motion)
    // to filter out static regions
    // applies to the prob image of each slice
    void MotionFilter (Series *stack, Config const &config); 
    void ProbFilter (Study *study, Config const &config); 
    void FindSquare (cv::Mat &mat, cv::Rect *bbox, Config const &config);

    struct Volume {
        float mean;
        float var;
        Volume (): mean(0), var(0) {
        }
    };

    void FindMinMaxVol (Study const &study, Volume *minv, Volume *maxv, Config const &config);

    static inline void report (std::ostream &os, Slice const &s, cv::Rect const &bound) {
        float r = std::sqrt(s.pred.area())/2 * s.meta.spacing;
        cv::Point_<float> raw_pt((bound.x + s.pred.x + s.pred.width/2.0) * s.meta.spacing / s.meta.raw_spacing,
                         (bound.y + s.pred.y + s.pred.height/2.0) * s.meta.spacing / s.meta.raw_spacing);
        float raw_r = r / s.meta.raw_spacing;
        os << s.path.native() << '\t' << r
             << '\t' << raw_pt.x << '\t' << raw_pt.y << '\t' << raw_r
             << '\t' << s.meta.raw_spacing
             << '\t' << s.meta.trigger_time;
        for (unsigned i = 0; i < Meta::SERIES_FIELDS; ++i) {
            os << '\t' << s.meta[i];
        }
        os << std::endl;
    }

    class Eval {
        static constexpr unsigned CASES = 500;
        float volumes[CASES][2];
        static float crps (float v, vector<float> const &x);
    public:
        static constexpr unsigned VALUES = 600;
        Eval ();
        float get (unsigned n1, unsigned n2) const {
            return volumes[n1-1][n2];
        }
        float score (fs::path const &, vector<std::pair<string, float>> *);
        float score (unsigned n1, unsigned n2, vector<float> const &x);
    };

    static inline double max_R (cv::Point2f const &p, cv::Rect_<float> const &r) {
        return std::max({cv::norm(p - r.tl()),
                        cv::norm(p - r.br()),
                        cv::norm(p - cv::Point2f(r.x + r.width, r.y)),
                        cv::norm(p - cv::Point2f(r.x, r.y + r.height))});
    }

    static inline void linearPolar (cv::Mat image,
                      cv::Mat *out,
                      cv::Point_<float> O, float R, int flags = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS) {
        IplImage tmp_in = image;
        IplImage *tmp_out = cvCreateImage(cvSize(image.cols, image.rows), IPL_DEPTH_32F, 1);
        CvPoint2D32f center;
        center.x = O.x;
        center.y = O.y;
        cvLinearPolar(&tmp_in, tmp_out, center, R, flags);
        *out = cv::Mat(tmp_out, true);
        /*
        cv::Mat out;
        cv::normalize(polar, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imwrite("xxx.png", out);
        */
        cvReleaseImage(&tmp_out);
    }
}

