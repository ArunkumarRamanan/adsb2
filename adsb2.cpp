#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/xml_parser.hpp>
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

}
