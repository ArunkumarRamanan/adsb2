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
            slice->polar_contour.clear();
            cv::Mat image = slice->polar_prob;
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
        CA1 (Config const &conf)
            : margin(conf.get<int>("adsb2.ca1.margin", 5)),
            thr(conf.get<float>("adsb2.ca1.th", 0.3)),
            smooth(conf.get<float>("adsb2.ca1.smooth", 200)),
            wall(conf.get<float>("adsb2.ca1.wall", 300))
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

    CA *make_ca_1 (Config const &config) {
        return new CA1(config);
    }

    struct Task {
        cv::Point_<float> C;
        float R;
        Slice *slice;
    };

    void study_CA1 (Study *study, Config const &config) {
        // compute bouding box
        vector<Task> tasks;
        for (Series &ss: *study) {
            cv::Rect_<float> lb = ss.front().pred_box;
            cv::Rect_<float> ub = lb;
            for (auto &s: ss) {
                cv::Rect_<float> r = unround(s.pred_box);
                lb &= r;
                ub |= r;
            }
            Task task;
            task.C = cv::Point2f(lb.x + 0.5 * lb.width, lb.y + 0.5 * lb.height);
            task.R = max_R(task.C, ub) * 3;
            if (lb.width == 0) task.R = 0;
            for (auto &s: ss) {
                task.slice = &s;
                tasks.push_back(task);
            }
        }
        string contour_model = config.get("adsb2.caffe.contour_model", (home_dir/fs::path("contour_model")).native());
        CA1 ca1(config);
#pragma omp parallel
        {
            Detector *det;
#pragma omp critical
            det = make_caffe_detector(contour_model);
            CHECK(det) << " cannot create detector.";
#pragma omp for schedule(dynamic, 1)
            for (unsigned i = 0; i < tasks.size(); ++i) {
                auto &task = tasks[i];
                Slice &slice = *task.slice;
                if (task.R == 0) {
                    slice.pred_area = 0;
                    continue;
                }
                slice.update_polar(task.C, task.R, det);
                ca1.apply_slice(&slice);
                if (slice.polar_contour.empty()) {
                    slice.pred_area = 0;
                    continue;
                }
                auto const &cc = slice.polar_contour;
                CHECK(cc.size() == slice.image.rows);
                cv::Mat polar(slice.image.size(), CV_32F, cv::Scalar(0));
                for (int y = 0; y < polar.rows; ++y) {
                    float *row = polar.ptr<float>(y);
                    for (int x = 0; x < cc[y]; ++x) {
                        row[x] = 1;
                    }
                }
                cv::Mat vis(slice.image.size(), CV_32F, cv::Scalar(0));
                for (int i = 1; i < cc.size(); ++i) {
                    cv::line(vis, cv::Point(cc[i-1], i-1), cv::Point(cc[i], i), cv::Scalar(-0xFF), 2);
                }

                cv::Mat cart;
                linearPolar(polar, &cart, task.C, task.R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
                //linearPolar(polar, &cart, task.C, task.R, CV_INTER_NN+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
                slice.pred_area = cv::sum(cart)[0];
                linearPolar(vis, &cart, task.C, task.R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);
                slice._extra = slice.image + cart;
            }
#pragma omp critical
            delete det;
        }
    }

}
