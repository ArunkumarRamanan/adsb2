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
#include "adsb2-cv.h"

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

    extern fs::path home_dir;
    void GlobalInit (char const *path, Config const &config);
    void dicom_setup (char const *path, Config const &config);
    cv::Mat load_dicom (fs::path const &, Meta *);

    class Slice;

    enum {
        ANNO_NONE = 0,
        ANNO_BOX = 1,
        ANNO_CIRCLE = 2,
        ANNO_ECLIPSE = 3,   // rotated 
        ANNO_POLY = 4
    };

    // annotations
    class AnnoOps {
    public:
        virtual int type () const = 0;
        virtual void load (Slice *, string const *) const = 0;
        virtual void shift (Slice *slice, cv::Point_<float> const &) const = 0;
        virtual void scale (Slice *slice, float rate) const = 0;
        virtual void fill (Slice const &, cv::Mat *, cv::Scalar const &) const = 0;
    };

    class BoxAnnoOps: public AnnoOps {
    public:
        typedef cv::Rect_<float> Data;
        virtual int type () const {
            return ANNO_BOX;
        }
        virtual void load (Slice *slice, string const *txt) const;
        virtual void shift (Slice *slice, cv::Point_<float> const &) const;
        virtual void scale (Slice *slice, float rate) const;
        virtual void fill (Slice const &, cv::Mat *, cv::Scalar const &) const;
    };

    class PolyAnnoOps {
    public:
        struct Data {
        };
        virtual int type () const {
            return ANNO_POLY;
        }
        virtual void load (Slice *, string const *) const;
        virtual void shift (Slice *slice, cv::Point_<float> const &) const;
        virtual void scale (Slice *slice, float rate) const;
        virtual void fill (Slice const &, cv::Mat *, cv::Scalar const &) const;
    };

    struct AnnoData {
        BoxAnnoOps::Data box; 
        PolyAnnoOps::Data poly;
    };

    extern BoxAnnoOps box_anno_ops;
    extern PolyAnnoOps poly_anno_ops;

    class Detector;

    struct Slice {
        int id;
        fs::path path;          // must always present
        Meta meta;              // available after load_raw
                                // spacing might be cooked, raw_spacing won't be cooked
        cv::Mat raw;            // raw image as loaded from DICOM, U16C1

        // size of image,vimage,label and prob must be the same when present
        cv::Mat image;          // cooked image             CV_32FC1
        cv::Mat vimage;         // cooked variance image    CV_32FC1

        bool do_not_cook;
        // the following fields are only available for annotated images
        string line;            // line as read from list file
        AnnoOps const *anno;    // points to global variable,
                                // should never be released
        AnnoData anno_data;

        // the following are prediction results
        cv::Mat prob;           // pixel-level prediction
        cv::Rect pred_box;      // bounding box prediction
        bool pred_box_reliable;

        cv::Point_<float> polar_C;   //
        float polar_R;              // available after update_polar is called
        cv::Mat polar_prob;         //
        vector<int> polar_contour;

        float pred_area;        // area prediction

        cv::Mat _label;         // temporarily used by import.cpp
        cv::Point_<float> _import_C; // nothing should be assumed for this field
        float _import_R;

        Slice ()
            : do_not_cook(false),
            anno(nullptr),
            pred_box(-1,-1,0,0),
            pred_box_reliable(false),
            pred_area(-1) {
        }

        Slice (string const &line);

        void clone (Slice *s) const; 

        void load_raw () {
            raw = load_dicom(path, &meta);
#if 1   // Yuanfang's annotation assume images are all in landscape position
            if (raw.rows > raw.cols) {
                cv::transpose(raw, raw);
            }
#endif
            image = raw;
        }

        void update_polar (cv::Point_<float> const &C, float R, Detector *);
        /*
        void update_polar (Detector *det) {
            cv::Rect_<float> r = unround(pred_box);
            update_polar(r, det);
        }
        */

#if 0
        cv::Mat roi () {
            CHECK(box.x >=0 && box.y >= 0);
            cv::Rect roi = round(box);
            return image(roi);
        }

        void eval (cv::Mat mat, float *s1, float *s2) const;
#endif
        // visualization should be the last step
        // after visualization, no further computation should be carried out
        void visualize (bool show_prob);
    };

    class Series: public vector<Slice> {
        fs::path path;
        bool sanity_check (bool fix = false);
        friend class Study;
    public:
        Series (){}
        // load from a directory of DCM files
        Series (fs::path const &, bool load = true, bool check = true, bool fix = false);

        // shrink all slices in the series to given bounding box
        void shrink (cv::Rect const &bb); 

        fs::path dir () const {
            return path;
        }

        /*
        cv::Size shape () const {
            return at(0).image.size();
        }

        void convert (int rtype, double alpha = 1, double beta = 0) {
            for (auto &s: *this) {
                s.image.convertTo(s.image, rtype, alpha, beta);
            }
        }
        */

        void visualize (bool show_prob = true);
        void save_dir (fs::path const &dir, fs::path const &ext); 
        void save_gif (fs::path const &path); 

        void getVImage (cv::Mat *);
    };

    class Study: public vector<Series> {
        fs::path path;
        bool sanity_check (bool fix = false);
        void check_regroup ();  // some times its necessary to regroup one series into
                                // multiple series
    public:
        Study ();
        // load from a directory of DCM files
        Study (fs::path const &, bool load = true, bool check = true, bool fix = false);
        fs::path dir () const {
            return path;
        }
        void shrink (cv::Rect const &bb) {
            for (auto &s: *this) {
                s.shrink(bb);
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
        void apply (Slice *) const;
        void apply (Series *) const;
        void apply (Study *) const;
    };

    // samples do not belong to a single directory
    class Slices: public vector<Slice>
    {
    public:
        Slices (fs::path const &list_path, fs::path const &root, Cook const &cook);
    };

    // A detector 
    class Detector {
    public:
        virtual ~Detector () {}
        virtual void apply (cv::Mat image, cv::Mat *prob) = 0;
    };

    Detector *make_caffe_detector (string const &path);
    /*
    Detector *make_cascade_detector (Config const &);
    Detector *make_scd_detector (Config const &);
    */

    class Sampler {
        // regular
        float max_color;
        std::uniform_real_distribution<float> delta_color; //(min_R, max_R);
        float max_linear_angle;
        float max_linear_scale;
        std::uniform_real_distribution<float> linear_angle;
        std::uniform_real_distribution<float> linear_scale;
        // polar
        std::uniform_real_distribution<float> polar_R; //(min_R, max_R);
        // shift polar center by polar_C * R at direction phi
        std::uniform_real_distribution<float> polar_C; //(0, max_C);
        std::uniform_real_distribution<float> polar_phi; //(0, M_PI * 2);
        int polar_kernel_size;
        cv::Mat polar_kernel;
        // = cv::Mat::ones(mk, mk, CV_8U);

        std::default_random_engine e;
    public:
        Sampler (Config const &config)
            : max_color(config.get<float>("adsb2.aug.color", 20)),
            delta_color(-max_color, max_color),
            max_linear_angle(config.get<float>("adsb2.aug.angle", 10.0 * M_PI / 180)),
            max_linear_scale(config.get<float>("adsb2.aug.scale", 1.2)),
            linear_angle(-max_linear_angle, max_linear_angle),
            linear_scale(-max_linear_scale, max_linear_scale),
            polar_R(config.get<float>("adsb2.aug.min_polar_R", 0.75),
                    config.get<float>("adsb2.aug.max_polar_R", 1.5)),
            polar_C(0, config.get<float>("adsb2.aug.max_polar_C", 0.4)),
            polar_phi(0, M_PI * 2),
            polar_kernel_size(config.get<int>("adsb2.aug.polar_kernel", 3)),
            polar_kernel(cv::Mat::ones(polar_kernel_size, polar_kernel_size, CV_8U))
        {
        }

        void linear (cv::Mat from_image,
                    cv::Mat from_label,
                    cv::Mat *to_image,
                    cv::Mat *to_label, bool no_perturb = false) {
            if (no_perturb) {
                *to_image = from_image;
                *to_label = from_label;
                return;
            }
            float color = delta_color(e);
            float angle = linear_angle(e);
            float scale = std::exp(linear_scale(e));
            bool flip((e() % 2) == 1);
            cv::Mat image, label;
            if (flip) {
                cv::flip(from_image, image, 1);
                cv::flip(from_label, label, 1);
            }
            else {
                image = from_image;
                label = from_label;
            }
            cv::Mat rot = cv::getRotationMatrix2D(cv::Point(image.cols/2, image.rows/2), angle, scale);
            cv::warpAffine(image, *to_image, rot, image.size());
            cv::warpAffine(label, *to_label, rot, label.size(), cv::INTER_NEAREST); // cannot interpolate labels
            *to_image += color;
        }

        void polar (cv::Mat from_image,
                          cv::Mat from_label,
                          cv::Point_<float> C,
                          float R,
                          cv::Mat *to_image,
                          cv::Mat *to_label,
                          bool no_perturb = false) {
            // randomization
            float color = 0;
            bool flip = false;
            if (!no_perturb) {
                float cr = polar_C(e) * R;  // center perturb
                float phi = polar_phi(e);
                C.x += cr * std::cos(phi);
                C.y += cr * std::sin(phi);
                R *= polar_R(e);
                color = delta_color(e);
                flip = ((e() % 2) == 1);
            }
            linearPolar(from_image, to_image, C, R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
            linearPolar(from_label, to_label, C, R, CV_INTER_NN+CV_WARP_FILL_OUTLIERS);
            /*
            imageF.convertTo(image, CV_8UC1);
            cv::equalizeHist(image, image);
            labelF.convertTo(label, CV_8UC1);
            */
            *to_image += color;
            cv::morphologyEx(*to_label, *to_label, cv::MORPH_CLOSE, polar_kernel);
            if (flip) {
                cv::flip(*to_image, *to_image, 0);
                cv::flip(*to_label, *to_label, 0);
            }
        }
    };

#if 0
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
                CHECK(sample.anno > 0);
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
#endif

    class CA {
    public:
        ~CA () {
        }
        virtual int level () const {
            return 0;
        }
        virtual void apply (Series *) const = 0;
        virtual void apply (Study *ss) const {
            for (Series &s: *ss) {
                apply(&s);
            }
        }
    };

    CA *make_ca_1 (Config const &);
    CA *make_ca_2 (Config const &);
    CA *make_ca_3 (Config const &);



    // use variance image (big variance == big motion)
    // to filter out static regions
    // applies to the prob image of each slice
    void Bound (Detector *det, Study *study, cv::Rect *box, Config const &config);
    void MotionFilter (Series *stack, Config const &config); 
    void ProbFilter (Study *study, Config const &config); 
    void FindSquare (cv::Mat &mat, cv::Rect *bbox, Config const &config);
    void study_CA1 (Study *, Config const &config);

    struct Volume {
        float mean;
        float var;
        Volume (): mean(0), var(0) {
        }
    };

    void FindMinMaxVol (Study const &study, Volume *minv, Volume *maxv, Config const &config);

    static inline void report (std::ostream &os, Slice const &s, cv::Rect const &bound) {
        float r = std::sqrt(s.pred_area)/2 * s.meta.spacing;
        cv::Point_<float> raw_pt((bound.x + s.pred_box.x + s.pred_box.width/2.0) * s.meta.spacing / s.meta.raw_spacing,
                         (bound.y + s.pred_box.y + s.pred_box.height/2.0) * s.meta.spacing / s.meta.raw_spacing);
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
}

