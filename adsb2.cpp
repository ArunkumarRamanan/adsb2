#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/xml_parser.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include "adsb2.h"

namespace adsb2 {
    void LoadConfig (string const &path, Config *config) {
        try {
            boost::property_tree::read_xml(path, *config);
        }
        catch (...) {
            LOG(WARNING) << "Cannot load config file " << path << ", using defaults.";
        }
    }

    void SaveConfig (string const &path, Config const &config) {
        boost::property_tree::write_xml(path, config);
    }

    void OverrideConfig (std::vector<std::string> const &overrides, Config *config) {
        for (std::string const &D: overrides) {
            size_t o = D.find("=");
            if (o == D.npos || o == 0 || o + 1 >= D.size()) {
                std::cerr << "Bad parameter: " << D << std::endl;
                BOOST_VERIFY(0);
            }
            config->put<std::string>(D.substr(0, o), D.substr(o + 1));
        }
    }

    void Sample::eval (cv::Mat mat, float *s1, float *s2) const {
        cv::Rect rect;
        round(box, &rect);
        cv::Mat roi = mat(rect);
        float total = cv::sum(mat)[0];
        float covered = cv::sum(roi)[0];
        /*
        cv::Scalar avg, stv;
        cv::meanStdDev(roi, avg, stv);
        */
        *s1 = covered / total;
        //*s2 = stv[0] / avg[0];

        *s2 = std::sqrt(roi.total()) * meta.spacing;
    }

    cv::Mat ImageLoader::load (string const &path, Meta *pmeta) const {
        cv::Mat raw;
        Meta meta;
        raw = load_raw(path, &meta);
        if (!raw.data) {
            return raw;
        }

        if (spacing > 0) {
            //float scale = spacing / meta.spacing;
            meta.spacing = spacing;
            float scale = meta.raw_spacing / meta.spacing;
            cv::Size sz(std::round(raw.cols * scale),
                        std::round(raw.rows * scale));
            cv::resize(raw, raw, sz);
        }

        if (raw.rows > raw.cols) {
            cv::transpose(raw, raw);
        }

        if (pmeta) *pmeta = meta;

        return raw;
    }

    bool ImageLoader::load (Sample *sample) const {
        Meta meta;
        cv::Mat image = load(sample->path, &meta);;
        if (!image.data) {
            sample->image = cv::Mat();
            return false;
        }

        if (meta.spacing != meta.raw_spacing) {
            //float scale = spacing / meta.spacing;
            sample->box *= meta.raw_spacing / meta.spacing;
        }
        sample->meta = meta;
        sample->image = image;
        return true;
    }

    void ImageLoader::load (string const &path, string const &root, vector<Sample> *samples) const {
        ifstream is(path.c_str());
        CHECK(is) << "Cannot open list file: " << path;
        Sample s;
        int id = 0;
        string line;
        while (getline(is, line)) {
            s.id = id;
            if (!s.parse(line)) {
                LOG(ERROR) << "Bad line: " << line;
                continue;
            }
            s.path = root + s.path;
            if (!fs::is_regular_file(fs::path(s.path))) {
                LOG(ERROR) << "Cannot find regular file: " << s.path;
                continue;
            }
            if (!load(&s)) {
                LOG(ERROR) << "Fail to load file: " << s.path;
                continue;
            }
            samples->emplace_back();
            std::swap(s, samples->back());
            ++id;
        }
        LOG(INFO) << "Loaded " << samples->size() << " samples.";
    };

    void DcmStack::getAvgStdDev (cv::Mat *avg, cv::Mat *stddev) {
        namespace ba = boost::accumulators;
        typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::count, ba::tag::variance, ba::tag::moment<2>>> Acc;
        CHECK(size());
        BOOST_VERIFY(at(0).type() == CV_16U);

        cv::Size shape = at(0).size();
        unsigned pixels = at(0).total();

        cv::Mat mu(shape, CV_32F);
        cv::Mat sigma(shape, CV_32F);
        vector<Acc> accs(pixels);
        for (auto const &image: *this) {
            uint16_t const *v = image.ptr<uint16_t const>(0);
            for (auto &acc: accs) {
                acc(*v);
                ++v;
            }
        }
        float *m = mu.ptr<float>(0);
        float *s = sigma.ptr<float>(0);
        //float *sp = spread.ptr<float>(0);
        for (auto const &acc: accs) {
            *m = ba::mean(acc);
            *s = std::sqrt(ba::variance(acc));
            //cout << *s << endl;
            //*sp = ba::max(acc) - ba::min(acc);
            ++m; ++s; //++sp;
        }
        *avg = mu;
        *stddev = sigma;
    }

    void DcmStack::getColorRange (ColorRange *range, float th) {
        cv::Mat mu, sigma;
        getAvgStdDev(&mu, &sigma);
        th = percentile<float>(sigma, th);
        vector<vector<int>> picked(sigma.rows);
        for (int i = 0; i < sigma.rows; ++i) {
            auto &v = picked[i];
            float const *p = sigma.ptr<float const>(i);
            for (int j = 0; j < sigma.cols; ++j) {
                if (p[j] >= th) {
                    v.push_back(j);
                }
            }
        }
        uint16_t low = std::numeric_limits<uint16_t>::max();
        uint16_t high = 0;
        uint16_t ulow = low;
        uint16_t uhigh = high;

        for (auto const &image: *this) {
            for (int i = 0; i < sigma.rows; ++i) {
                auto const &v = picked[i];
                uint16_t const *p = image.ptr<uint16_t const>(i);
                for (int j = 0; j < sigma.cols; ++j) {
                    if (p[j] > high) high = p[j];
                    if (p[j] < low) low = p[j];
                }
                for (int j: v) {
                    if (p[j] > uhigh) uhigh = p[j];
                    if (p[j] < ulow) ulow = p[j];
                }
            }
        }
        CHECK(low <= ulow);
        CHECK(high >= uhigh);
        range->min = low;
        range->max = high;
        range->umin = ulow;
        range->umax = uhigh;
    }

    void ImageAdaptor::apply (cv::Mat *to, ColorRange const &range, uint8_t tlow, uint8_t thigh) {
        cv::Mat from = to->clone();
        CHECK(from.type() == CV_16UC1);
        to->create(from.size(), CV_8UC1);
        // [low, ulow) -> [0, tlow)
        // [ulow, uhigh] -> [tlow, thigh]
        // (uhigh, high] -> (thigh, 255]
        int low = range.min;
        int ulow = range.umin;
        int high = range.max;
        int uhigh = range.umax;


        if (ulow - low < tlow) {
            // e.g.   min = 100, low = 101
            //                   tlow = 10
            // we'll lower min = 101-10 = 91, so there's
            // 1-1 correspondance between [91, 101) and [0, 10)
            low = ulow - tlow;
        }
        if (high - uhigh < 255 - thigh) {
            // e.g.   high = 100, max = 101
            //        thigh = 245,
            // we raise max = 100 + 255 - 245 = 110
            //        [100, 110] <-> [245, 255]
            high = uhigh + (255 - thigh);

        }
        for (int i = 0; i < from.rows; ++i) {
            uint16_t const *f = from.ptr<uint16_t const>(i);
            uint8_t *t = to->ptr<uint8_t>(i);
            for (int j = 0; j < from.cols; ++j) {
                uint16_t x = f[j];
                int y = 0;
                if (x < low) {
                    y = 0;
                }
                if (x < ulow) {
                    // [low, ulow) -> [0, tlow)
                    y = (x - low) * tlow / (ulow - low);
                }
                else if (x <= uhigh) {
                    // [ulow, uhigh] -> [tlow, thigh]
                    y = (x - ulow) * (thigh - tlow) / (uhigh - ulow) + tlow;
                    // x == ulow  ==> tlow
                    // x == uhigh ==> thigh
                }
                else if (x <= high) {
                    // (uhigh, high] -> (thigh, 255]
                     y = (x - uhigh) * (255 - thigh) / (high - uhigh) + thigh;
                    // uhigh =>  thigh
                    // high => 255
                }
                else {
                    y = 255;
                }
                CHECK(y >= 0 && y <= 255);
                t[j] = uint8_t(y);
            }
        }
    }
}
