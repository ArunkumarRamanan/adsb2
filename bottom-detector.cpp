#include <xgboost_wrapper.h>
#include "adsb2.h"

namespace adsb2 {

    extern fs::path model_dir;
    class BottomDetectorImpl: public Classifier {
        BoosterHandle cfier;
    public:
        BottomDetectorImpl (fs::path const &path) {
            int r = XGBoosterCreate(NULL, 0, &cfier);
            CHECK(r == 0); 
            CHECK(cfier);
            r = XGBoosterLoadModel(cfier, path.native().c_str());
            CHECK(r == 0) << "Failed to load " << path; 
        }
        ~BottomDetectorImpl () {
            XGBoosterFree(cfier);
        }
        virtual float apply (vector<float> const &ft) const {
            //array<float, SL_SIZE> const &data) const {
            //vector<float> ft{data[SL_BSCORE], data[SL_PSCORE], data[SL_CSCORE], data[SL_CCOLOR], data[SL_ARATE]};
            DMatrixHandle dmat;
            int r = XGDMatrixCreateFromMat(&ft[0], 1, ft.size(), 0, &dmat);
            bst_ulong len;
            float const *out;
            XGBoosterPredict(cfier, dmat, 0, 0, &len, &out);
            return out[0];
        }
    };

    Classifier *make_xgboost_classifier (fs::path const &path) {
        return new BottomDetectorImpl(path);
    }
}
