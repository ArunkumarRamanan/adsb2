
    class Gaussian {
        cv::Mat mean;
        cv::Mat cov;
        cv::Mat icov;
    public:
        Gaussian (cv::Mat samples, cv::Mat weights)
            : mean(1, samples.cols, CV_32F, cv::Scalar(0)),
            cov(samples.cols, samples.cols, CV_32F, cv::Scalar(0))
        {
            CHECK(samples.rows == weights.rows);
            CHECK(weights.cols == 1);
            float sum = 0;
            for (int i = 0; i < samples.rows; ++i) {
                float w = weights.ptr<float>(i)[0];
                mean += w * samples.row(i);
                sum += w;
            }
            mean /= sum;
            for (int i = 0; i < samples.rows; ++i) {
                cv::Mat row = samples.row(i) - mean;
                cov += weights.ptr<float>(i)[0] * row.t() * row;
            }
            cov /= sum;
            icov = cov.inv(cv::DECOMP_SVD);
            LOG(INFO) << "mean: " << mean;
            LOG(INFO) << "cov: " << cov;
            LOG(INFO) << "icov: " << icov;
        }
        cv::Mat prob (cv::Mat in) const {
            CHECK(in.cols == mean.cols);
            cv::Mat r(in.rows, 1, CV_32F);
            float *ptr = r.ptr<float>(0);
            for (int i = 0; i < in.rows; ++i) {
                cv::Mat r = in.row(i) - mean;
                cv::Mat k = r * icov * r.t();
                CHECK(k.rows == 1);
                CHECK(k.cols == 1);
                float v = std::exp(-0.5 * k.ptr<float>(0)[0]);
                ptr[i] = v;
            }
            return r;
        }
    };

    void Var2Prob (cv::Mat oin, cv::Mat *out, float pth, int mk) {
        cv::Mat in;
        pth = percentile<float>(oin, pth);
        cv::threshold(oin, in, pth, 1.0, cv::THRESH_BINARY);
        cv::Mat kernel = cv::Mat::ones(mk, mk, CV_32F);
        cv::morphologyEx(in, in, cv::MORPH_OPEN, kernel);
        cv::Mat tmp = in.mul(oin);
        in = tmp;


        cv::Mat s(in.total(), 2, CV_32F);
        {
            int o = 0;
            for (int i = 0; i < in.rows; ++i) {
                for (int j = 0; j < in.cols; ++j) {
                    float *ptr = s.ptr<float>(o++);
                    ptr[0] = i;
                    ptr[1] = j;
                }
            }
            CHECK(o == s.rows);
        }
        Gaussian g(s, in.reshape(1, s.rows));
        cv::Mat p = g.prob(s);
        *out = p.reshape(1, in.rows);
        CHECK(out->size() == in.size());
    }

