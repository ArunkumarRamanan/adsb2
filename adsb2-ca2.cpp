#include <boost/multi_array.hpp>
#include "adsb2.h"

namespace adsb2 {
    class CA2: public CA {
        float thr;
        float smooth;
        float wall;
        float penalty (int dx) const {
            return smooth *abs(dx);
        };

        void helper (Slice *slice) const {
        }
    public:
        CA2 (Config const &conf)
            : thr(conf.get<float>("adsb2.ca1.th", 0.3)),
            smooth(conf.get<float>("adsb2.ca1.smooth", 200)),
            wall(conf.get<float>("adsb2.ca1.wall", 300))
        {
        }
        void apply_slice (Slice *s) {
        }
        void apply (Series *ss) const {
        }
    };

    void study_CA2 (Study *study, Config const &config) {
    }

}
