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
        *s2 = roi.total();
    }
}
