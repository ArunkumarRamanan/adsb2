#pragma once

// common opencv data structure operations
namespace adsb2 {

    using std::vector;

    template <typename T>
    struct Circle_ {
        T x, y, r;
    };

    typedef Circle_<int> Circle;

    static inline cv::Rect round (cv::Rect_<float> const &f) {
        int x = std::round(f.x);
        int y = std::round(f.y);
        int width = std::round(f.x + f.width) - x;
        int height = std::round(f.y + f.height) - y;
        return cv::Rect(x, y, width, height);
    }

    static inline cv::Point_<float> unround (cv::Point const &r) {
        return cv::Point_<float>(r.x, r.y);
    }

    static inline cv::Rect_<float> unround (cv::Rect const &r) {
        return cv::Rect_<float>(r.x, r.y, r.width, r.height);
    }

    static inline cv::Rect_<float> cscale (cv::Rect_<float> const &r, float scale) {
        float cx = r.x + r.width/2;
        float cy = r.y + r.height/2;
        float w = r.width * scale;
        float h = r.height * scale;
        return cv::Rect_<float>(cx - w/2, cy - h/2, w, h);
    }

    template <typename T>
    static inline cv::Rect_<float>  operator * (cv::Rect_<T> &f, float scale) {
        return cv::Rect_<float>(f.x * scale,
                                f.y * scale,
                                f.width * scale,
                                f.height * scale);
    }

    static inline cv::Size round (cv::Size_<float> const &sz) {
        return cv::Size(std::round(sz.width), std::round(sz.height));
    }

    static inline cv::Point round (cv::Point_<float> const &pt) {
        return cv::Point(std::round(pt.x), std::round(pt.y));
    }

    template <typename T>
    static inline cv::Size_<float> operator * (cv::Size_<T> const &sz, float scale) {
        return cv::Size_<float>(sz.width * scale, sz.height * scale);
    }

    template <typename I>
    void bound (I begin, I end, int *b, float margin) {
        float cc = 0;
        I it = begin;
        while (it < end) {
            cc += *it;
            if (cc >= margin) break;
            ++it;
        }
        *b = it - begin;
    }

    static inline void bound (vector<float> const &v, int *x, int *w, float margin) {
        int b, e;
        bound(v.begin(), v.end(), &b, margin);
        bound(v.rbegin(), v.rend(), &e, margin);
        e = v.size() - e;
        if (e < b) e = b;
        *x = b;
        *w = e - b;
    }

    float accumulate (cv::Mat const &image, vector<float> *X, vector<float> *Y);

    static inline void bound (cv::Mat const &image, cv::Rect *rect, float th) {
        vector<float> X;
        vector<float> Y;
        float total = accumulate(image, &X, &Y);
        float margin = total * (1.0 - th) / 2;
        bound(X, &rect->x, &rect->width, margin);
        bound(Y, &rect->y, &rect->height, margin);
    }

    template <typename T>
    void percentile (vector<T> &all, vector<float> const &p, vector<T> *v) {
        sort(all.begin(), all.end());
        v->resize(p.size());
        for (unsigned i = 0; i < p.size(); ++i) {
            int t = int(std::floor(all.size() * p[i]));
            if (t < 0) t = 0;
            if (t >= all.size()) t = all.size() - 1;
            v->at(i) = all[t];
        }
    }

    template <typename T>
    void percentile (cv::Mat const &mat, vector<float> const &p, vector<T> *v) {
        vector<T> all;
        CHECK(mat.total());
        for (int i = 0; i < mat.rows; ++i) {
            T const *p = mat.ptr<T const>(i);
            for (int j = 0; j < mat.cols; ++j) {
                all.push_back(p[j]);
            }
        }
        percentile(all, p, v);
    }

    template <typename T>
    T percentile (cv::Mat const &mat, float p) {
        vector<T> v;
        percentile(mat, vector<float>{p}, &v);
        return v[0];
    }


    static inline double max_R (cv::Point2f const &p, cv::Rect_<float> const &r) {
        return std::max({cv::norm(p - r.tl()),
                        cv::norm(p - r.br()),
                        cv::norm(p - cv::Point2f(r.x + r.width, r.y)),
                        cv::norm(p - cv::Point2f(r.x, r.y + r.height))});
    }

    static inline void linearPolar (cv::Mat from,
                      cv::Mat *out,
                      cv::Point_<float> O, float R, int flags = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS) {
        cv::Mat image;
        if (from.type() == CV_32F) {
            image = from;
        }
        else {
            from.convertTo(image, CV_32F);
        }
        IplImage tmp_in = image;
        IplImage *tmp_out = cvCreateImage(cvSize(image.cols, image.rows), IPL_DEPTH_32F, 1);
        CvPoint2D32f center;
        center.x = O.x;
        center.y = O.y;
        cvLinearPolar(&tmp_in, tmp_out, center, R, flags);
        cv::Mat to = cv::Mat(tmp_out, true);
        if (from.type() == CV_32F) {
            *out = to;
        }
        else {
            to.convertTo(*out, from.type());
        }
        /*
        cv::Mat out;
        cv::normalize(polar, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imwrite("xxx.png", out);
        */
        cvReleaseImage(&tmp_out);
    }

    static inline void vconcat3 (cv::Mat a, cv::Mat b, cv::Mat c, cv::Mat *out) {
        cv::Mat tmp;
        cv::vconcat(a, b, tmp);
        cv::vconcat(tmp, c, *out);
    }
    static inline void hconcat3 (cv::Mat a, cv::Mat b, cv::Mat c, cv::Mat *out) {
        cv::Mat tmp;
        cv::hconcat(a, b, tmp);
        cv::hconcat(tmp, c, *out);
    }
    static inline void type_convert (cv::Mat *v, int t) {
        cv::Mat tmp;
        v->convertTo(tmp, t);
        *v = tmp;
    }

    static inline void bound_box (cv::Mat const &m, cv::Rect *bb) {
        CHECK(m.type() == CV_32F);
        int min_x = m.cols;
        int max_x = -1;
        int min_y = m.rows;
        int max_y = -1;
        for (int y = 0; y < m.rows; ++y) {
            float const *row = m.ptr<float const>(y);
            for (int x = 0; x < m.cols; ++x) {
                if (row[x]) { 
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);
                }
            }
        }
        if ((min_x > max_x) || (min_y > max_y)) *bb = cv::Rect();
        else {
            bb->x = min_x;
            bb->y = min_y;
            bb->width = max_x - min_x + 1;
            bb->height = max_y - min_y + 1;
        }
    }

    static inline void loop_check (cv::Mat &m, uint16_t) {
        CHECK(m.type() == CV_16U);
    }

    static inline void loop_check (cv::Mat &m, float) {
        CHECK(m.type() == CV_32F);
    }

    static inline void loop_check (cv::Mat &m, uint8_t) {
        CHECK(m.type() == CV_8U);
    }

    template<typename T, typename F>
    static inline void loop (cv::Mat &m, F f) {
        loop_check(m, T());
        for (int i = 0; i < m.rows; ++i) {
            T *p = m.ptr<T>(i);
            for (int j = 0; j < m.cols; ++j) {
                f(p[j]);
            }
        }
    }

    template<typename T1, typename T2, typename F>
    static inline void loop (cv::Mat &m1, cv::Mat &m2, F f) {
        loop_check(m1, T1());
        loop_check(m2, T2());
        CHECK(m1.rows == m2.rows);
        CHECK(m1.cols == m2.cols);
        for (int i = 0; i < m1.rows; ++i) {
            T1 *p1 = m1.ptr<T1>(i);
            T2 *p2 = m2.ptr<T2>(i);
            for (int j = 0; j < m1.cols; ++j) {
                f(p1[j], p2[j]);
            }
        }
    }

    static inline float box_score (cv::Mat &prob, cv::Rect const &box) {
        float total = cv::sum(prob)[0];
        float inside = cv::sum(prob(box))[0];
        float outside = total - inside;
        float ba = box.area();
        return 1.0 * (ba - inside + outside) / ba;
    }

    void draw_text (cv::Mat &img, std::string const &text, cv::Point org, int line = 0, cv::Scalar color = cv::Scalar(0xFF));

    static constexpr int GRAYS = 256;
    static inline void scale_color(cv::Mat *img, float lb, float ub) {
        loop<float>(*img, [lb, ub](float &v) {
            v = std::round((v - lb) * GRAYS / (ub - lb));
            if (v < 0) v = 0;
            else if (v >= GRAYS) v =  GRAYS - 1;
        });
    }

    cv::Point_<float> weighted_box_center (cv::Mat &prob, cv::Rect box);
}
