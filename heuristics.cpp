#include <queue>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include "spline.h"
#include "adsb2.h"

namespace adsb2 {
    using std::queue;

    struct InterpEntry {
        bool good;
        cv::Scalar v;

        void pack (float area, cv::Rect const &box) {
            v[0] = area;
            // radius
            v[1] = std::sqrt(box.width * box.width
                             + box.height * box.height) / 2;
            // cx
            v[2] = box.x + 0.5 * box.width;
            // cy
            v[3] = box.y + 0.5 * box.height;
        }

        void unpack (float *area, cv::Rect *box) {
            *area = v[0];
            float r = v[1];
            *box = round(cv::Rect_<float>(v[2]-r, v[3]-r, 2*r, 2*r));
        }
    };

    class Interp: public vector<InterpEntry> {
        static int constexpr EXT = 3;   // for circular
        template <int dim>
        void fill (vector<double> *v, int ext) {
            v->clear();
            int sz = size();
            for (int i = sz - ext; i < sz; ++i) {
                if (!at(i).good) continue;
                if (dim < 0) {
                    v->push_back(i - sz);
                }
                else {
                    v->push_back(at(i).v[dim]);
                }
            }
            for (int i = 0; i < sz; ++i) {
                if (!at(i).good) continue;
                if (dim < 0) {
                    v->push_back(i);
                }
                else {
                    v->push_back(at(i).v[dim]);
                }
            }
            for (int i = 0; i < ext; ++i) {
                if (!at(i).good) continue;
                if (dim < 0) {
                    v->push_back(i + sz);
                }
                else {
                    v->push_back(at(i).v[dim]);
                }
            }
        }
        template <int dim>
        void estimate (vector<double> const &X, int ext) {
            vector<double> Y;
            fill<dim>(&Y, ext);
            tk::spline s;
            s.set_points(X, Y);
            for (int i = 0; i < size(); ++i) {
                if (!at(i).good) {
                    at(i).v[dim] = s(i + ext);
                }
            }
        }
    public:
        void run (bool circular) {
            CHECK(size() >= EXT);
            vector<double> X;
            int ext = circular ? EXT: 0;
            fill<-1>(&X, ext);
            estimate<0>(X, ext);
            estimate<1>(X, ext);
            estimate<2>(X, ext);
            estimate<3>(X, ext);
        }
    };
    /*
    class InterBox: public vector<InterBoxEntry> {
    public:
        enum {
            MODE_TIME;
            MODE_SPACE;
        };
        void run (bool mode) {
            CHECK(viable);
            if (mode == MODE_TIME) {
                std::swap(left, right);
            }
            else if (mode == MODE_SPACE) {
                right[0] = right[1] = 0;
            }
            cv::Scalar lb = left;
            unsigned next = 0;
            while (next < size()) {
                if (at(next).good) {
                    lb = 
                }
            }
            
        }
    };
    */

#if 0
    void Bound (Detector *det, Study *study, cv::Rect *box, Config const &config) {
        float ext = config.get<float>("adsb2.bound.ext", 4);
        if (study->size() < 3) return;
        float best_v = -1;
        unsigned best = 0;
        for (unsigned i = 0; i < study->size(); ++i) {
            float v = cv::sum(study->at(i).front().vimage)[0];
            if (v > best_v) {
                best_v = v;
                best = i;
            }
        }
        best = (study->size() + 1) / 2;
        Series &mid = study->at(best);
        Series ss;
        ss.resize(1);
        mid[0].clone(&ss[0]);
        det->apply(ss[0].image, &ss[0].prob);
        MotionFilter(&ss, config);
        FindSquare(ss[0].prob, &ss[0].box, config);
        cv::Rect bb = round(cscale(unround(ss[0].box), ext));
        if (bb.x < 0) bb.x = 0;
        if (bb.y < 0) bb.y = 0;
#pragma omp parallel for
        for (unsigned i = 0; i < study->size(); ++i) {
            study->at(i).shrink(bb);
        }
        *box = bb;
    }
#endif

    static int conn_comp (cv::Mat *mat, cv::Mat const &weight, vector<float> *cnt) {
        // return # components
        CHECK(mat->type() == CV_8UC1);
        CHECK(mat->isContinuous());
        CHECK(weight.type() == CV_32F);
        cv::Mat out(mat->size(), CV_8UC1, cv::Scalar(0));
        CHECK(out.isContinuous());
        uint8_t const *ip = mat->ptr<uint8_t const>(0);
        uint8_t *op = out.ptr<uint8_t>(0);
        float const *wp = weight.ptr<float const>(0);

        int c = 0;
        int o = 0;
        if (cnt) cnt->clear();
        for (int y = 0; y < mat->rows; ++y)
        for (int x = 0; x < mat->cols; ++x) {
            do {
                if (op[o]) break;
                if (ip[o] == 0) break;
                // find a new component
                ++c;
                queue<int> todo;
                op[o] = c;
                todo.push(o);
                float W = 0;
                while (!todo.empty()) {
                    int t = todo.front();
                    todo.pop();
                    W += wp[t];
                    // find neighbors of t and add
                    int tx = t % mat->cols;
                    int ty = t / mat->cols;
                    for (int ny = std::max(0, ty-1); ny <= std::min(mat->rows-1,ty+1); ++ny) 
                    for (int nx = std::max(0, tx-1); nx <= std::min(mat->cols-1,tx+1); ++nx) {
                        // (ny, ix) is connected
                        int no = t + (ny-ty) * mat->cols + (nx-tx);
                        if (op[no]) continue;
                        if (ip[no] == 0) continue;
                        op[no] = c;
                        todo.push(no);
                    }
                }
                if (cnt) cnt->push_back(W);
            } while (false);
            ++o;
        }
        *mat = out;
        CHECK(c == cnt->size());
        return c;
    }

    void MotionFilter (Series *pstack, Config const &config) {
        Series &stack = *pstack;
        float bin_th = config.get<float>("adsb2.mf.bin_th", 0.8);
        float supp_th = config.get<float>("adsb2.mf.supp_th", 0.6);
        int dilate = config.get<int>("adsb2.mf.dilate", 10);

        cv::Mat p(stack.front().images[IM_IMAGE].size(), CV_32F, cv::Scalar(0));
        for (auto &s: stack) {
            p = cv::max(p, s.images[IM_PROB]);
        }
        cv::normalize(p, p, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::threshold(p, p, 255 * bin_th, 255, cv::THRESH_BINARY);
        vector<float> cc;
        conn_comp(&p, stack.front().images[IM_VAR], &cc);
        CHECK(cc.size());
        float max_c = *std::max_element(cc.begin(), cc.end());
        for (unsigned i = 0; i < cc.size(); ++i) {
            if (cc[i] < max_c * supp_th) cc[i] = 0;
        }
        for (int y = 0; y < p.rows; ++y) {
            uint8_t *ptr = p.ptr<uint8_t>(y);
            for (int x = 0; x < p.cols; ++x) {
                uint8_t c = ptr[x];
                if (c == 0) continue;
                --c;
                CHECK(c < cc.size());
                ptr[x] = cc[c] ? 1: 0;
            }
        }
        cv::Mat kernel = cv::Mat::ones(dilate, dilate, CV_8U);
        cv::dilate(p, p, kernel);
        cv::Mat np;
        p.convertTo(np, CV_32F);
#pragma omp parallel for
        for (unsigned i = 0; i < stack.size(); ++i) {
            auto &s = stack[i];
            cv::Mat prob = s.images[IM_PROB].mul(np);
            s.images[IM_BFILTER] = np;
            s.images[IM_PROB] = prob;
        }
        // find connected components of p
    }

    void ProbFilter (Study *study, Config const &config) {
        float bin_th = config.get<float>("adsb2.pf.bin_th", 0.8);
        float supp_th = config.get<float>("adsb2.pf.supp_th", 0.6);
        int dilate = config.get<int>("adsb2.pf.dilate", 10);

        cv::Mat p(study->front().front().images[IM_IMAGE].size(), CV_32F, cv::Scalar(0));
        cv::Mat pv(study->front().front().images[IM_IMAGE].size(), CV_32F, cv::Scalar(0));
        vector<Slice *> slices;
        for (auto &s: *study) {
            for (auto &ss: s) {
                slices.push_back(&ss);
                p = cv::max(p, ss.images[IM_PROB]);
            }
            pv = cv::max(pv, s.front().images[IM_VAR]);
        }
        cv::normalize(p, p, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::threshold(p, p, 255 * bin_th, 255, cv::THRESH_BINARY);
        vector<float> cc;
        conn_comp(&p, pv, &cc);
        CHECK(cc.size());
        float max_c = *std::max_element(cc.begin(), cc.end());
        for (unsigned i = 0; i < cc.size(); ++i) {
            if (cc[i] < max_c * supp_th) cc[i] = 0;
        }
        for (int y = 0; y < p.rows; ++y) {
            uint8_t *ptr = p.ptr<uint8_t>(y);
            for (int x = 0; x < p.cols; ++x) {
                uint8_t c = ptr[x];
                if (c == 0) continue;
                --c;
                CHECK(c < cc.size());
                ptr[x] = cc[c] ? 1: 0;
            }
        }
        cv::Mat kernel = cv::Mat::ones(dilate, dilate, CV_8U);
        cv::dilate(p, p, kernel);
        cv::Mat np;
        p.convertTo(np, CV_32F);
#pragma omp parallel for
        for (unsigned i = 0; i < slices.size(); ++i) {
            cv::Mat prob = slices[i]->images[IM_PROB].mul(np);
            slices[i]->images[IM_BFILTER] = np;
            slices[i]->images[IM_PROB] = prob;
        }
        // find connected components of p
    }

    void gen_candidate (cv::Mat const &mat, cv::Rect const &r,
                        cv::Rect *c) {   // at most four candidate
        c[0].width = c[1].width = c[2].width = c[3].width = r.width + 1;
        c[0].height = c[1].height = c[2].height = c[3].height = r.height + 1;

        c[0].x = r.x;   c[0].y = r.y;
        c[1].x = r.x;   c[1].y = r.y-1;
        c[2].x = r.x-1; c[2].y = r.y;
        c[3].x = r.x-1; c[3].y = r.y-1;
        for (int i = 0; i < 4; ++i) {
            auto &cc = c[i];
            if ((cc.x < 0) || (cc.y < 0) 
                || (cc.x + cc.width > mat.cols)
                || (cc.y + cc.height > mat.rows)) {
                cc.width = cc.height = -1;
            }
        }
    }

    class IntegrateImage {
        cv::Mat s;
    public:
        IntegrateImage (cv::Mat const &image) 
            : s(image.rows + 1, image.cols + 1, CV_32F, cv::Scalar(0))
        {
            vector<float> acc(image.cols, 0);
            for (int y = 0; y < image.rows; ++y) {
                float const *from = image.ptr<float const>(y);
                float *to = s.ptr<float>(y+1);
                for (int x = 0; x < image.cols; ++x) {
                    acc[x] += from[x];
                    to[x+1] = to[x] + acc[x];
                }
            }
        }
        float sum (cv::Rect const &rect) const {
            float const *row1 = s.ptr<float const>(rect.y);
            float const *row2 = s.ptr<float const>(rect.y + rect.height);
            return row2[rect.x + rect.width] + row1[rect.x]
                 - row2[rect.x] - row1[rect.x + rect.width];
                 
        }
    };

    void FindSquare (cv::Mat &mat, cv::Rect *rect, Config const &config) {
        float c_th = config.get<float>("adsb2.square.cth", 0.85); // probability cap
        float r_th = config.get<float>("adsb2.square.rth", 0.95) * M_PI/4;
        float b_th = config.get<float>("adsb2.square.bth", 0.75);
        CHECK(mat.type() == CV_32F);
        cv::Mat probs;
        cv::normalize(mat, probs, 0, 1, cv::NORM_MINMAX, CV_32F);
        probs /= c_th;
        cv::threshold(probs, probs, 1, 1, cv::THRESH_TRUNC);
        IntegrateImage I(probs);
        cv::Rect seed;
        bound(mat, &seed, b_th);
        // find the best square within rect
        vector<cv::Rect> cc;
        if (seed.width <= seed.height) {
            cc.resize(seed.height + 1 - seed.width);
            cc[0] = seed;
            cc[0].height = cc[0].width;
            for (unsigned i = 1; i < cc.size(); ++i) {
                cc[i] = cc[0];
                cc[i].y += i;
            }
        }
        else {
            cc.resize(seed.width + 1 - seed.height);
            cc[0] = seed;
            cc[0].width = cc[0].height;
            for (unsigned i = 1; i < cc.size(); ++i) {
                cc[i] = cc[0];
                cc[i].x += i;
            }
        }
        float best = -1;
        for (auto const &c: cc) {
            float s = I.sum(c);
            if (s > best) {
                seed = c;
                best = s;
            }
        }
        for (;;) {
            cc.resize(4);
            gen_candidate(probs, seed, &cc[0]);
            bool updated = false;
            for (auto const &c: cc) {
                if (c.width < 0 || c.height < 0) continue;  // out-of-bound rects
                float s = I.sum(c);
                float r = s / c.area();
                if ((s > best) && (r >= r_th)) {
                    seed = c;
                    best = s;
                    updated = true;
                }
            }
            if (!updated) break;
        }
        CHECK(best > 0);
        *rect = seed;
    }

    void FindBox (Slice *slice, Config const &conf) {
        cv::Mat prob = slice->images[IM_PROB];
        FindSquare(prob, &slice->box, conf);
        slice->data[SL_BSCORE] = box_score(prob, slice->box);
    }

    // if X and Y are independent normal variables
    // X + Y is also normal variable
    // mean(X + Y) = mean(X) + mean(Y)
    // var(X + Y) = var(X) + var(Y)

    struct SeriesVolume {
        Volume min, max;
        float location;
    };

    float calc_volume (float a, float b, float d) {
        return d * (a + b + std::sqrt(a * b)) / 3;
    }

    void acc_volume (Volume *v, float a, float b, float d) {
        v->mean += d * (a + b + std::sqrt(a * b)) / 3;
        v->coef1 += (sqrt(a) + sqrt(b)) * d;
        v->coef2 += d;
    }

    void FindMinMaxVol (Study const &study, Volume *minv, Volume *maxv, Config const &config) {
        // steps
        int W = config.get<int>("adsb2.smooth.W", 3);
        CHECK(W >= 1);
        vector<SeriesVolume> seriesV;
        for (auto const &series: study) {
            SeriesVolume v;
            v.location = series.front().meta.slice_location;
            vector<float> r;
            for (auto const &s: series) {
                //CHECK(s.area >= 0);
                CHECK(s.meta.slice_location == v.location);
                r.push_back(s.data[SL_AREA] * s.meta.spacing * s.meta.spacing);
            }
            for (unsigned j = 0; j < W; ++j) { // extend the range for smoothing
                r.push_back(r[j]);
            }
            // smooth r
            for (unsigned j = 0; j + W <= r.size(); ++j) {
                namespace ba = boost::accumulators;
                typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::variance>> Acc;
                Acc acc;
                for (unsigned w = 0; w < W; ++w) {
                    acc(r[j+w]);
                }
                Volume sv;
                sv.mean = ba::mean(acc);
                sv.var = ba::variance(acc);
                if (j == 0) {
                    v.min = sv;
                    v.max = sv;
                }
                else {
                    if (sv.mean < v.min.mean) {
                        v.min = sv;
                    }
                    if (sv.mean > v.max.mean) {
                        v.max = sv;
                    }
                }
            }
            //CHECK(v.min.mean >= 0);
            //CHECK(v.max.mean >= 0);
            seriesV.push_back(v);
        }
        for (unsigned i = 1; i < seriesV.size() - 1; ++i) {
            if (seriesV[i].min.mean == 0) {
                seriesV[i].min.mean = std::sqrt(seriesV[i-1].min.mean
                                              * seriesV[i+1].min.mean);

            }
            if (seriesV[i].max.mean == 0) {
                seriesV[i].max.mean = std::sqrt(seriesV[i-1].max.mean
                                              * seriesV[i+1].max.mean);
            }
        }
        // accumulate
        Volume min;
        Volume max;
        for (unsigned i = 1; i < seriesV.size(); ++i) {
            auto const &a = seriesV[i-1];
            auto const &b = seriesV[i];
            float gap = b.location - a.location;
            CHECK(gap > 0);
            acc_volume(&min, a.min.mean, b.min.mean, gap);
            acc_volume(&max, a.max.mean, b.max.mean, gap);
        }
        *minv = min;
        *maxv = max;
    }

    void ComputeContourProb (Study *study, Config const &conf)
    {
        for (Series &ss: *study) {
            /*
            cv::Rect_<float> ub = ss.front().box;
            for (auto &s: ss) {
                cv::Rect_<float> r = unround(s.box);
                ub |= r;
            }
            */
            for (auto &s: ss) {
                if (s.box.width == 0) {
                    s.data[SL_AREA] = 0;
                    s.images[IM_POLAR] = cv::Mat();
                }
                else {
                    cv::Rect_<float> lb = unround(s.box);
                    //cv::Point2f C = cv::Point2f(lb.x + 0.5 * lb.width, lb.y + 0.5 * lb.height);
                    cv::Point_<float> C = weighted_box_center(s.images[IM_PROB], s.box);
                    float R = max_R(C, lb) * 3;
                    s.update_polar(C, R);
                }
            }
        }
        ApplyDetector("contour", study, IM_POLAR, IM_POLAR_PROB, 1.0, study->front().front().images[IM_IMAGE].rows/4);
    }

#if 0
    void RefinePolarBound (Study *study, Config const &config) {
        float fill_rate = config.get("adsb2.refine.fill_rate", M_PI/4 * 0.85);
        float good_slice_rate = config.get("adsb2.refine.good_slice_rate", 0.75);
        // evaluate healthness of bound and contour
        vector<int> bads;
        vector<cv::Rect> ubs;
        for (unsigned i = 0; i < study->size(); ++i) {
            Series &ss = study->at(i);
            cv::Rect lb(cv::Point(0, 0), ss[0].image.size());
            cv::Rect ub;
            float max_area = 0;
            // bad slice: contour is not detected, area is 0, or area/polar_bb_area < fill_rate
            // rest are good slice
            int good = 0;
            for (Slice &s: ss) {
                if (s.area <= 0) continue; // bad anyway
                if (s.area / s.polar_box.area() < fill_rate) {
                    s.area = 0;
                    continue;
                }
                ++good;
                lb = lb & s.polar_box;
                ub = ub | s.polar_box;
            }
            ubs.push_back(ub);
            if ((lb.area() == 0) || (good < (ss.size() * good_slice_rate))) {
                // whole slice is bad
                bads.push_back(i);
                for (auto &s: ss) { // invalidate everything, rely on findMinMax to interpolate
                    s.area = 0;
                    s.box = cv::Rect();
                }
            }
            else {
                // use the good ones
                // interpolate the bad ones
                // !! TODO
                Interp interp;
                interp.resize(ss.size());
                for (unsigned j = 0; j < ss.size(); ++j) {
                    Slice &s = ss[j];
                    auto &w = interp[j];
                    if (s.area > 0) {  // good
                        w.good = true;
                        w.pack(s.area, s.box);
                    }
                    else {
                        w.good = false;
                    }
                }
                interp.run(true);
                for (unsigned j = 0; j < ss.size(); ++j) {
                    Slice &s = ss[j];
                    auto &w = interp[j];
                    if (!w.good) {  // interp value
                        w.unpack(&s.area, &s.box);
                    }
                }
            }
        }
        if (bads.empty()) return;
        // if the series is good, we try to save the bad slices by interpolating
        // then we try to save the bad series by interpolating
        // if this doesn't do, then just fail
        Interp interp;  // interpolate bounding box
        interp.resize(ubs.size());
        for (unsigned i = 0; i < interp.size(); ++i) {
            auto &w = interp[i];
            w.good = true;
            w.pack(0, ubs[i]);
        }
        for (unsigned i: bads) {
            interp[i].good = false;
        }
        interp.run(false);
        // extract interpolations
        for (unsigned i: bads) {
            float dummy;
            cv::Rect box;
            interp[i].unpack(&dummy, &box);
            for (Slice &s: study->at(i)) {
                s.box = box;
            }
        }
    }
#endif

    void ComputeTop (Study *study, Config const &conf) {
        float th = conf.get("adsb2.top.th", 0.2);
        //config.put("adsb2.caffe.model", "model2");
        for (unsigned sid = 0; sid < study->size(); ++sid) {
            auto &slices = study->at(sid);
            std::cerr << "Computing top probablity of " << slices.size() << "  slices..." << std::endl;
            boost::progress_display progress(slices.size(), std::cerr);
            int bad = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+:bad)
            for (unsigned i = 0; i < slices.size(); ++i) {
                Detector *det = Detector::get("top");
                CHECK(det) << " cannot create detector.";
                vector<float> prob(2);
                det->apply(slices[i].images[IM_IMAGE], &prob);
                float p = prob[1];
                slices[i].data[SL_TSCORE] = p;
                if (p > th) {
                    ++bad;
                }
#pragma omp critical
                ++progress;
            }
            if (bad == 0) break;
        }
    }

    void RefineTop (Study *study, Config const &conf) {
        vector<float> tops(study->size());
        float min = 0;
        for (unsigned i = 0; i < tops.size(); ++i) {
            float sum = 0;
            int c = 0;
            for (auto &s: study->at(i)) {
                sum += s.data[SL_TSCORE];
                ++c;
            }
            sum /= c;
            if (sum < min) min = sum;
            tops[i] = sum;
        }
        int cut = 0;
        if (min > 0.2) cut = 1;
        for (unsigned i = 0; i < tops.size(); ++i) {
            if (tops[i] > 0.5) cut = i + 1;
            else break;
        }
        for (unsigned i = 0; i < cut; ++i) {
            for (auto &s: study->at(i)) {
                s.data[SL_AREA] = 0;
            }
        }
    }

    void PatchBottomBoundHelper (Slice *s, Config const &conf) {
        float mag = conf.get<float>("adsb2.patch_bb.mag", 2.0);
        cv::Mat image;
        s->images[IM_RAW].convertTo(image, CV_32F);
        cv::Size sz = s->images[IM_IMAGE].size();
        cv::Size mag_sz = round(sz * mag);
        cv::resize(image, image, mag_sz);
        float lb = s->data[SL_COLOR_LB];
        float ub = s->data[SL_COLOR_UB];
        scale_color(&image, lb, ub);

        // center of the box
        cv::Point C(s->box.x + s->box.width/2,
                     s->box.y + s->box.height/2);
        // center of the box in magnified image
        cv::Point mag_C(C.x * mag, C.y * mag);

        cv::Point roi_shift = mag_C - C;

        cv::Rect bb(roi_shift, sz);

        cv::Mat roi = image(bb).clone();
        cv::Mat roi_prob;
        // we only detect a ROI in enlarged image
        // the ROI has the same size as the slice's image
        // the offset is picked, such that the enlarged box's center is still at C
        // relative to the ROI
        Detector *det = Detector::get("bound");
        CHECK(det) << " cannot create detector.";
        det->apply(roi, &roi_prob);
        cv::Mat prob(image.size(), CV_32F, cv::Scalar(0));
        roi_prob.copyTo(prob(bb));
        cv::resize(prob, prob, s->images[IM_PROB].size());
        {
            cv::Mat bf = s->images[IM_BFILTER];
            CHECK(bf.data) << "ProbFilter must be invoked before patching bb";
            cv::Mat tmp = prob.mul(bf);
            prob = tmp;
        }
        cv::Rect box;
        FindSquare(prob, &box, conf);
        float bscore = box_score(prob, box);
        float orig_bscore = s->data[SL_BSCORE];
        if (bscore + 0.001 < orig_bscore) { // update new probability
            s->data[SL_BSCORE] = bscore;
            s->data[SL_BSCORE_DELTA] = orig_bscore - bscore;
            s->images[IM_PROB] = prob;
            s->box = box;
        }
    }

    void PatchBottomBound(Study *study, Config const &conf) {
        float th = conf.get<float>("adsb2.patch_bb.th", 0.5);
        // only try to fix the bottom half
        vector<Slice *> todo;
        for (unsigned sr = 2 * study->size()/3;
                      sr < study->size(); ++sr) {
            for (Slice &s: study->at(sr)) {
                if (s.data[SL_BSCORE] > th) {
                    todo.push_back(&s);
                }
            }
        }
        LOG(WARNING) << "Patching " << todo.size() << " bottom slices...";
        boost::progress_display progress(todo.size(), std::cerr);
#pragma omp parallel for
        for (unsigned i = 0; i < todo.size(); ++i) {
            PatchBottomBoundHelper(todo[i], conf);
#pragma omp critical
            ++progress;
        }
    }

    void EvalBottom(Study *study, Config const &conf) {
        unsigned l = study->size() / 2;
        if (l < 1) l = 1;
        Classifier *det = Classifier::get("bottom");
        for (unsigned i = l; i < study->size(); ++i) {
            auto &top = study->at(i-1);
            auto &cur = study->at(i);
            if (top.size() == cur.size()) {
                for (unsigned i = 0; i < top.size(); ++i) {
                    cur[i].data[SL_ARATE] =
                        cur[i].data[SL_AREA] / top[i].data[SL_AREA];
                }
            }
            else {
                LOG(WARNING) << "sax size mismatch";
                for (unsigned i = 0; i < top.size(); ++i) {
                    cur[i].data[SL_ARATE] = 1.0;
                }
            }
            for (auto &s: cur) {
                vector<float> ft{s.data[SL_BSCORE],
                                 s.data[SL_PSCORE],
                                 s.data[SL_CSCORE],
                                 s.data[SL_CCOLOR],
                                 s.data[SL_ARATE]};
                s.data[SL_BOTTOM] = det->apply(ft);
            }
        }
    }

    void RefineBottomHelperSimple (vector<Slice *> &slices, Config const &conf) {
        int first_good = slices.size() - 4;
        int first_bad = slices.size() - 3;
        if (first_good < 0) return;
        while (first_bad < slices.size()) {
            if (slices.at(first_bad)->data[SL_BOTTOM] > 0.6) break;
            ++first_good;
            ++first_bad;
        }
        if (first_bad >= slices.size()) return;
        float init = sqrt(slices[first_good]->data[SL_AREA]);
        int n = slices.size() - first_good;
        for (int i = 1; i < n; ++i) {
            float area = init * (n - i - 1) / (n-1);
            area *= area;
            slices[first_good + i]->data[SL_AREA] = area;
            slices[first_good + i]->data[SL_BOTTOM_PATCH] = 1;
        }
    }

    void study_CA1 (Slice *pslice, Config const &config, bool vis);

    void RefineBottomHelper (vector<Slice *> &slices, Config const &conf) {
        for (unsigned x = 1; x < slices.size(); ++x) {
            Slice *prev = slices[x-1];
            Slice *cur = slices[x];
            cv::Rect box = prev->polar_box;
            // improve ?? center of top % white pixels
            cv::Point_<float> C = weighted_box_center(cur->images[IM_IMAGE], box);
            float R = max_R(C, box) * 3;
            cur->update_polar(C, R);
            ApplyDetector("contour", cur, IM_POLAR, IM_POLAR_PROB, 1.0, cur->images[IM_IMAGE].rows/4);
            // study ca1
            study_CA1(cur, conf, true);
            cv::Rect inter = cur->polar_box & box;
            cur->data[SL_BOTTOM_PATCH] = 1;
            if (cur->data[SL_AREA] == 0) {
                // if this one fail, all the rest should fail, too
                for (unsigned x1 = x; x1 < slices.size(); ++x1) {
                    slices[x1]->polar_box = cv::Rect();
                    slices[x1]->data[SL_AREA] = 0;
                    slices[x1]->data[SL_BOTTOM_PATCH] = 2;
                }
                return;
            }
        }
    }

    void RefineBottom (Study *study, Config const &conf) {
        float max_area = conf.get<float>("adsb2.refine.max", 300);
        unsigned l = study->size() / 2;
        unsigned ns = study->at(l).size();
        for (unsigned i = l+1; i < study->size(); ++i) {
            if (ns != study->at(i).size()) return;
        }
        std::cerr << "Refining bottoms..." << std::endl;
        boost::progress_display progress(ns, std::cerr);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned sid = 0; sid < ns; ++sid) {
            vector<Slice *> slices;
            for (unsigned j = 0; j < study->size(); ++j) {
                slices.push_back(&study->at(j)[sid]);
            }
            RefineBottomHelperSimple(slices, conf);
#pragma omp critical
            ++progress;
        }
    }


}

