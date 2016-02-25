#include <limits>
#include <boost/multi_array.hpp>
#include "adsb2.h"

namespace adsb2 {
    class CA1: public CA {
        struct E {
            float opt;   // optimal value
            int prev;    // prev slice
            int prev0;   // optimal location of 0, for circular opt
        };
        typedef boost::multi_array<E, 2> WorkSpaceBase;
        class WorkSpace:public WorkSpaceBase {
        public:
            WorkSpace (int rows, int cols): WorkSpaceBase(boost::extents[rows][cols]) {
            }
        };
        float smooth;
        float th;
        int extra_delta;
        float penalty (int dx) const {
            return smooth *abs(dx);
        };

        void helper (Slice *slice) const {
            slice->polar_contour.clear();
            cv::Mat image = slice->images[IM_POLAR_PROB];
            CHECK(image.type() == CV_32F);
            // thr big => th small => tight

            WorkSpace ws(image.rows, image.cols);
            int best_cc = 0;
            for (int y = 0; y < image.rows; ++y) {
                CHECK(y < image.rows);
                E *e = &(ws[y][0]);
                float const *I = image.ptr<float const>(y);
                if (y == 0) {
                    float acc = 0;
                    for (int x = 0; x < image.cols; ++x) {
                        float delta = I[x] - th;
                        acc += delta;
                        e[x].opt = acc;
                        e[x].prev = -1;
                        e[x].prev0 = -1;
                    }
                    continue;
                }
                E *prev = &(ws[y-1][0]);
                float acc = 0;
                best_cc = 0;
                float best_cc_score = -1;
                for (int x = 0; x < image.cols; ++x) {
                    float delta = I[x] - th;
                    acc += delta;
                    int lb = std::max(x - 7, 0);
                    int ub = std::min(x + 7, image.cols-1);
                    float best_score = -1;
                    int best_prev = 0;
                    for (int p = lb; p <= ub; ++p) {
                        float score = prev[p].opt + acc - penalty(p - x);
                        if (y + 1 == image.rows) {  // need to consider connection to 0
                            score -= penalty(prev[p].prev0 -x);
                        }
                        if (score > best_score) {
                            best_score = score;
                            best_prev = p;
                        }
                    }
                    e[x].opt = best_score;
                    e[x].prev = best_prev;
                    if (y == 1) {
                        e[x].prev0 = e[x].prev;
                    }
                    else {
                        e[x].prev0 = prev[best_prev].prev0;
                    }
                    if ((x == 0) || (best_score > best_cc_score)) {
                        best_cc_score = best_score;
                        best_cc = x;
                    }
                }
            }
            int y = image.rows - 1;
            vector<int> seg;
            while (y >= 0) {
                seg.push_back(best_cc);
                best_cc = ws[y][best_cc].prev;
                --y;
            }
            CHECK(best_cc = -1);
            std::reverse(seg.begin(), seg.end());
            slice->polar_contour.swap(seg);
        }

        float contour_avg (Slice *slice, int delta) const {
            float sum = 0;
            auto const &ctr = slice->polar_contour;
            cv::Mat const &image = slice->images[IM_POLAR];
            CHECK(ctr.size() == image.rows);
            for (unsigned i = 0; i < ctr.size(); ++i) {
                float const *ptr = image.ptr<float const>(i);
                int x = ctr[i] + delta;
                if (x < 0) x = 0;
                if (x >= image.cols) x = image.cols - 1;
                sum += ptr[x];
            }
            return sum / ctr.size();
        }

        void extend (Slice *slice) const {
            static const int L = 10;
            vector<float> dist(2*L+1);
            float sum = 0;
            for (int i = -L; i <= L; ++i) {
                sum += dist[L+i] = contour_avg(slice, i);
            }
            float best = -std::numeric_limits<float>::max();
            int best_nleft = 0;
            float left = 0;
            for (unsigned nleft = 1; nleft < dist.size(); ++nleft) {
                left += dist[nleft-1];
                float right = sum - left;
                float nright = dist.size() - nleft;

                float gap = left / nleft - right / nright;
                if (gap > best) {
                    best = gap;
                    best_nleft = nleft;
                }
            }
            int delta = best_nleft - L;
            if (delta < 0) delta = 0;
            delta += extra_delta;
            if (delta > 0) {
                for (auto &p: slice->polar_contour) {
                    p += delta;
                }
            }

        }
        bool do_extend;
    public:
        CA1 (Config const &conf)
            : th(conf.get<float>("adsb2.ca1.th", 0.5)),
            smooth(conf.get<float>("adsb2.ca1.smooth", 3)),
            extra_delta(conf.get<int>("adsb2.ca1.extra", 0)),
            do_extend(conf.get<int>("adsb2.ca1.extend", 0) > 0)
              
        {
        }
        void apply_slice (Slice *s) {
            helper(s);
            if (do_extend) {
                extend(s);
            }
        }
        void apply (Series *ss) const {
#pragma omp parallel for
            for (unsigned i = 0; i < ss->size(); ++i) {
                helper(&ss->at(i));
                if (do_extend) {
                    extend(&ss->at(i));
                }
            }
        }
    };

    void study_CA1 (Slice *pslice, Config const &config, bool vis) {
        CA1 ca1(config);
        Slice &slice = *pslice;
        if (!slice.images[IM_POLAR_PROB].data) {
            slice.polar_box = cv::Rect();
            return;
        }
        ca1.apply_slice(&slice);
        if (slice.polar_contour.empty()) {
            slice.data[SL_AREA] = 0;
            return;
        }
        auto const &cc = slice.polar_contour;
        CHECK(cc.size() == slice.images[IM_IMAGE].rows);
        cv::Mat polar(slice.images[IM_IMAGE].size(), CV_32F, cv::Scalar(0));
        for (int y = 0; y < polar.rows; ++y) {
            float *row = polar.ptr<float>(y);
            for (int x = 0; x < cc[y]; ++x) {
                row[x] = 1;
            }
        }

        linearPolar(polar, &slice.images[IM_LABEL], slice.polar_C, slice.polar_R, CV_INTER_NN+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);

        bound_box(slice.images[IM_LABEL], &slice.polar_box);

        {
            static int constexpr ext = 5;
            cv::Rect box = slice.polar_box;
            box.x -= ext;
            box.y -= ext;
            box.width += ext * 2;
            box.height += ext * 2;
            box = box & cv::Rect(cv::Point(0,0), slice.images[IM_IMAGE].size());

            cv::Mat inside;
            linearPolar(polar, &inside, slice.polar_C, slice.polar_R, CV_INTER_NN+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
            cv::Mat mask = inside(box).clone();
            type_convert(&mask, CV_8U);
            cv::Mat color = slice.images[IM_IMAGE](box).clone();
            float cs1, ps1; // inside
            color_sum(color, mask, &cs1, &ps1);
            CHECK(cs1 >= 0);
            if (ps1 <= 0) {
                LOG(ERROR) << "ps1 == 0: " << ps1;
                ps1 = 1;
            }
            CHECK(ps1 > 0);

            cv::Mat kernel = cv::Mat::ones(ext, ext, CV_8U);
            cv::dilate(mask, mask, kernel);
            float cs2, ps2; // outside
            color_sum(color, mask, &cs2, &ps2);
            cs2 -= cs1;
            ps2 -= ps1;
            CHECK(cs2 >= 0);
            if (ps2 <= 0) {
                LOG(ERROR) << "ps2 <= 0: " << ps2;
                ps2 = 1;
            }
            CHECK(ps2 > 0);

            cs1 /= ps1;
            cs2 /= ps2;
            slice.data[SL_CCOLOR] = cs1 - cs2;
        }
        

        slice.data[SL_AREA] = cv::sum(slice.images[IM_LABEL])[0];
        slice.data[SL_PSCORE] = box_score(slice.images[IM_LABEL], slice.polar_box);
        slice.data[SL_CSCORE] = box_score(slice.images[IM_LABEL], slice.box);

        if (vis) {
            cv::Mat vis_cart;
            cv::Mat vis(slice.images[IM_IMAGE].size(), CV_32F, cv::Scalar(0));
            for (int i = 1; i < cc.size(); ++i) {
                cv::line(vis, cv::Point(cc[i-1], i-1), cv::Point(cc[i], i), cv::Scalar(-0xFF), 2);
            }
            linearPolar(vis, &vis_cart, slice.polar_C, slice.polar_R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);
            hconcat3(slice.images[IM_POLAR] + vis, slice.images[IM_POLAR_PROB] * 255 + vis, slice.images[IM_IMAGE] + vis_cart, &slice._extra);
        }
    }

    void study_CA1 (Study *study, Config const &config, bool vis) {
        // compute bouding box
        vector<Slice *> tasks;
        study->pool(&tasks);
        CA1 ca1(config);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < tasks.size(); ++i) {
            study_CA1(tasks[i], config, vis);
        }
    }

}
