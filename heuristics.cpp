#include <queue>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include "adsb2.h"

namespace adsb2 {
    using std::queue;

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
        Series &mid = study->at(best);
        Series ss;
        ss.resize(1);
        mid[0].clone(&ss[0]);
        det->apply(&ss[0]);
        MotionFilter(&ss, config);
        FindSquare(ss[0].prob, &ss[0].pred, config);
        cv::Rect bb = ss[0].pred;
        int r = bb.width * ext - bb.width;
        bb.width += r;
        r /= 2;
        bb.x -= r;
        if (bb.x < 0) bb.x = 0;
        r = bb.height * ext - bb.height;
        bb.height += r;
        r /= 2;
        bb.y -= r;
        if (bb.y < 0) bb.y = 0;
#pragma omp parallel for
        for (unsigned i = 0; i < study->size(); ++i) {
            study->at(i).bound(bb);
        }
        *box = bb;
    }

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

        cv::Mat p(stack.front().image.size(), CV_32F, cv::Scalar(0));
        for (auto &s: stack) {
            p = cv::max(p, s.prob);
        }
        cv::normalize(p, p, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::threshold(p, p, 255 * bin_th, 255, cv::THRESH_BINARY);
        vector<float> cc;
        conn_comp(&p, stack.front().vimage, &cc);
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
            cv::Mat prob = s.prob.mul(np);
            s.prob = prob;
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

    // if X and Y are independent normal variables
    // X + Y is also normal variable
    // mean(X + Y) = mean(X) + mean(Y)
    // var(X + Y) = var(X) + var(Y)

    struct SeriesVolume {
        Volume min, max;
        float location;
    };

    void FindMinMaxVol (Study const &study, Volume *minv, Volume *maxv, Config const &config) {
        // steps
        int W = config.get<int>("adsb2.smooth.W", 3);
        CHECK(W >= 3);
        vector<SeriesVolume> seriesV;
        for (auto const &series: study) {
            vector<float> r;
            for (auto const &s: series) {
                r.push_back(s.pred.area() * s.meta.spacing * s.meta.spacing);
            }
            for (unsigned j = 0; j < W; ++j) { // extend the range for smoothing
                r.push_back(r[j]);
            }
            // smooth r
            SeriesVolume v;
            v.location = series.front().meta[Meta::SLICE_LOCATION];
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
            seriesV.push_back(v);
        }
        // accumulate
        Volume min;
        Volume max;
        for (unsigned i = 1; i < seriesV.size(); ++i) {
            auto const &a = seriesV[i-1];
            auto const &b = seriesV[i];
            float gap = b.location - a.location;
            CHECK(gap > 0);
            min.mean += (a.min.mean + b.min.mean) * gap/2;
            min.var += (a.min.var + b.min.var) * gap/2;
            max.mean += (a.max.mean + b.max.mean) * gap/2;
            max.var += (a.max.var + b.max.var) * gap/2;
        }
        *minv = min;
        *maxv = max;
    }

}

