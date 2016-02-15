#include <boost/multi_array.hpp>
#include "adsb2.h"

namespace adsb2 {
    class CA2: public CA {
        struct E {
            float opt;    
            int prev;    // prev slice
        };
        typedef boost::multi_array<E, 2> WorkSpaceBase;
        class WorkSpace:public WorkSpaceBase {
        public:
            WorkSpace (int rows, int cols): WorkSpaceBase(boost::extents[rows][cols]) {
            }
        };
        int margin;
        float thr;
        float smooth;
        float wall;
        float penalty (int dx) const {
            return smooth *abs(dx);
        };

        void helper (Slice *slice) const {
            slice->polar_contour.clear();
            cv::Mat image = slice->images[IM_POLAR_PROB];
            CHECK(image.type() == CV_32F);
            if (image.cols < margin * 2) return;
            float th = 0;
            {
                float left_mean = cv::mean(image.colRange(0, margin))[0];
                float right_mean = cv::mean(image.colRange(image.cols - margin, image.cols))[0];
                //cerr << left_mean << ' ' << right_mean << endl;
                if (!(right_mean < left_mean)) return;
                th = left_mean + (right_mean - left_mean) * thr;
            }

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
                        if (delta < 0) delta *= wall;
                        acc += delta;
                        e[x].opt = acc;
                        e[x].prev = -1;
                    }
                    continue;
                }
                E *prev = &(ws[y-1][0]);
                float acc = 0;
                best_cc = 0;
                float best_cc_score = -1;
                for (int x = 0; x < image.cols; ++x) {
                    float delta = I[x] - th;
                    if (delta < 0) delta *= wall;
                    acc += delta;
                    int lb = std::max(x - 7, 0);
                    int ub = std::min(x + 7, image.cols-1);
                    float best_score = -1;
                    int best_prev = 0;
                    for (int p = lb; p <= ub; ++p) {
                        float score = prev[p].opt + acc - penalty(p - x);
                        if (score > best_score) {
                            best_score = score;
                            best_prev = p;
                        }
                    }
                    e[x].opt = best_score;
                    e[x].prev = best_prev;
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
    public:
        CA2 (Config const &conf)
            : margin(conf.get<int>("adsb2.ca1.margin", 5)),
            thr(conf.get<float>("adsb2.ca1.th", 0.6)),
            smooth(conf.get<float>("adsb2.ca1.smooth", 150/255.0)),
            wall(conf.get<float>("adsb2.ca1.wall", 300/255.0))
        {
        }
        void apply_slice (Slice *s) {
            helper(s);
        }
        void apply (Series *ss) const {
#pragma omp parallel for
            for (unsigned i = 0; i < ss->size(); ++i) {
                helper(&ss->at(i));
            }
        }
    };

    void study_CA2 (Study *study, Config const &config, bool vis) {
        // compute bouding box
        vector<Slice *> tasks;
        study->pool(&tasks);
        CA2 ca1(config);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < tasks.size(); ++i) {
            Slice &slice = *tasks[i];
            if (!slice.images[IM_POLAR_PROB].data) {
                slice.polar_box = cv::Rect();
                slice.polar_score = 0;
                continue;
            }
            ca1.apply_slice(&slice);
            if (slice.polar_contour.empty()) {
                slice.area = 0;
                continue;
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

            linearPolar(polar, &slice.images[IM_LABEL], slice.polar_C, slice.polar_R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
            bound_box(slice.images[IM_LABEL], &slice.polar_box);
            slice.area = cv::sum(slice.images[IM_LABEL])[0];
            slice.data[SL_PSCORE] = box_score(slice.images[IM_LABEL], slice.polar_box);
            slice.data[SL_CSCORE] = box_score(slice.images[IM_LABEL], slice.box);

            if (vis) {
                cv::Mat vis_cart;
                cv::Mat vis(slice.images[IM_IMAGE].size(), CV_32F, cv::Scalar(0));
                for (int i = 1; i < cc.size(); ++i) {
                    cv::line(vis, cv::Point(cc[i-1], i-1), cv::Point(cc[i], i), cv::Scalar(-0xFF), 2);
                }
                linearPolar(vis, &vis_cart, slice.polar_C, slice.polar_R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);
                cv::hconcat(slice.images[IM_POLAR] + vis, slice.images[IM_IMAGE] + vis_cart, slice._extra);
            }
        }
    }

}
