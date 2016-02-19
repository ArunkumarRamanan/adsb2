#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <boost/assert.hpp>
namespace adsb2 {

    using std::string;
    using std::array;
    using std::vector;
    using std::ifstream;

    static inline double sqr (double x) {
        return x * x;
    }

    class EM {
    public:
        static unsigned constexpr N = 600;
        struct Sample {
            string sid;
            double target;
            double mu;
            double sigma;
            vector<double> cdf;

            void gaussianAcc () {
                cdf.resize(N);
                for (unsigned i = 0; i < N; ++i) {
                    double x = -(double(i) - mu) / sigma;
                    cdf[i] = 0.5*erfc(x);
                }
            }
        };
    private:
        double A, dA;
        array<double, N> B, dB;   // base

        struct WS {
            array<double, N> f;
            array<double, N> x;
            array<double, N> P;
            array<double, N> dHx;
            double X;
        };

        void prep (Sample const &s, WS *ws) {
            ws->X = 0;
            for (unsigned i = 0; i < N; ++i) {
                double v = (s.mu - i) / s.sigma;
                v *= v;
                ws->f[i] = v;
                v *= A;
                v += B[i];
                v = exp(v);
                ws->x[i] = v;
                ws->X += v;
            }
            double acc = 0;
            double sumPn2 = 0;
            std::fill(ws->dHx.begin(), ws->dHx.end(), 0);
            for (unsigned n = 0; n < N; ++n) {
                acc += ws->x[n];
                double Pn = ws->P[n] = acc / ws->X;

                sumPn2 += sqr(Pn);

                double t2 = 0, t3 = 0;
                if (n >= s.target) {
                    t2 = Pn;
                    t3 = -1;
                }

                for (unsigned i = 0; i <= n; ++i) {
                    ws->dHx[i] += Pn + t2 + t3;
                }
                for (unsigned i = n + 1; i < N; ++i) {
                    ws->dHx[i] += t2;
                }
            }
            for (unsigned n = 0; n < N; ++n) {
                ws->dHx[n] -= sumPn2;
                ws->dHx[n] /= ws->X;
            }
            // x_i = exp( a * f_i + b_i)
        }

        void forward (Sample *s) {
            WS ws;
            prep(*s, &ws);
            s->cdf = vector<double>(ws.P.begin(), ws.P.end());
        }

        void backward (Sample const &s) {
            WS ws;
            prep(s, &ws);

            dA = 0;
            // dA = sum_i  dH/dx_i * dx_i/dA
            // dB_i = dH/dx_i * dx_i / dB_i
            for (unsigned n = 0; n < N; ++n) {
                dA += ws.dHx[n] * ws.x[n] * ws.f[n];
                dB[n] = ws.dHx[n] * ws.x[n];
            }
        }

    public:
        EM () {
            A = -0.5;
            std::fill(B.begin(), B.end(), 0);
        }

        void load (string const &path);
        void save (string const &path);

        static void load (string const &path, vector<Sample> *samples) {
            ifstream is(path.c_str());
            BOOST_VERIFY(is);
            Sample s;
            samples->clear();
            while (is >> s.sid >> s.target >> s.mu >> s.sigma) {
                samples->push_back(s);
            }
        }

        void train (vector<Sample> const &ss) {
            for (auto &s: ss) {
                backward(s);
            }
        }

        void predict (vector<Sample> *ss) {
            for (auto &s: *ss) {
                forward(&s);
            }
        }
    };
}

