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

    static inline void linearPolar (cv::Mat image,
                      cv::Mat *out,
                      cv::Point_<float> O, float R, int flags = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS) {
        IplImage tmp_in = image;
        IplImage *tmp_out = cvCreateImage(cvSize(image.cols, image.rows), IPL_DEPTH_32F, 1);
        CvPoint2D32f center;
        center.x = O.x;
        center.y = O.y;
        cvLinearPolar(&tmp_in, tmp_out, center, R, flags);
        *out = cv::Mat(tmp_out, true);
        /*
        cv::Mat out;
        cv::normalize(polar, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imwrite("xxx.png", out);
        */
        cvReleaseImage(&tmp_out);
    }
}
