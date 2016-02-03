#include <boost/multi_array.hpp>
#include "adsb2.h"

namespace adsb2 {
    class CA1: public CA {
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
            cv::Mat image = slice->polar_prob;
            CHECK(image.type() == CV_32F);
            CHECK(image.cols >= margin *2);
            float th = 0;
            {
                float left_mean = cv::mean(image.colRange(0, margin))[0];
                float right_mean = cv::mean(image.colRange(image.cols - margin, image.cols))[0];
                //cerr << left_mean << ' ' << right_mean << endl;
                CHECK(right_mean < left_mean);
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
        CA1 (Config const &conf)
            : margin(conf.get<int>("adsb2.ca1.margin", 5)),
            thr(conf.get<float>("adsb2.ca1.th", 0.3)),
            smooth(conf.get<float>("adsb2.ca1.smooth", 200)),
            wall(conf.get<float>("adsb2.ca1.wall", 300))
        {
        }
        void apply (Series *ss) const {
#pragma omp parallel for
            for (unsigned i = 0; i < ss->size(); ++i) {
                helper(&ss->at(i));
            }
        }
    };

    CA *make_ca_1 (Config const &config) {
        return new CA1(config);
    }



}
