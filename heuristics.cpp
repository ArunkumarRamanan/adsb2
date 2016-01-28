#include <queue>
#include "adsb2.h"

namespace adsb2 {
    using std::queue;

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
}

