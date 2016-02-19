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
    using std::ofstream;

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
        static constexpr double LIMIT = 3; // 5 * sigma
        static constexpr unsigned QUANT = 50;
        array<double, QUANT> T, dT;

        struct WS {
            array<double, N> f;
            array<double, N> x;
            array<double, N> P;
            array<double, N> dHx;
            double X;
            double loss;
        };

        void prep (Sample const &s, WS *ws) {
            // x_i = exp(T[l])
            ws->X = 0;
            for (unsigned n = 0; n < N; ++n) {
                double l = abs(s.mu - n) * QUANT / (LIMIT * s.sigma);
                double v = -100;
                int lb = floor(l);
                int ub = lb + 1;
                if (ub < QUANT) {
                    v = T[lb] * (ub - l) + T[ub] * (l - lb);
                }
                else if (lb < QUANT) {
                    v = T[lb];
                }
                v = exp(v);
                ws->x[n] = v;
                ws->X += v;
            }
            double acc = 0;
            double sumPn2 = 0;
            ws->loss = 0;
            std::fill(ws->dHx.begin(), ws->dHx.end(), 0);
            for (unsigned n = 0; n < N; ++n) {
                acc += ws->x[n];
                double Pn = ws->P[n] = acc / ws->X;

                if (n < s.target) {
                    ws->loss += sqr(Pn);
                }
                else {
                    ws->loss += sqr(1-Pn);
                }

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
            ws->loss /= N;
            for (unsigned n = 0; n < N; ++n) {
                ws->dHx[n] -= sumPn2;
                ws->dHx[n] /= ws->X;
            }
        }

        double forward (Sample *s) {
            WS ws;
            prep(*s, &ws);
            s->cdf = vector<double>(ws.P.begin(), ws.P.end());
            return ws.loss;
        }

        double backward (vector<Sample> const &ss) {
            vector<WS> wss(ss.size());
            for (unsigned i = 0; i < ss.size(); ++i) {
                prep(ss[i], &wss[i]);
            }
            std::fill(dT.begin(), dT.end(), 0);
            double loss = 0;
            for (unsigned i = 0; i < ss.size(); ++i) {
                auto &s = ss[i];
                auto &ws = wss[i];
                loss += ws.loss;
                for (unsigned n = 0; n < N; ++n) {
                    /*
                        dT[l] += ws.dHx[n] * ws.x[n];
                        */
                    double l = abs(s.mu - n) * QUANT / (LIMIT * s.sigma);
                    int lb = floor(l);
                    int ub = lb + 1;
                    if (ub < QUANT) {
                        dT[lb] += ws.dHx[n] * ws.x[n] * (ub - l);
                        dT[ub] += ws.dHx[n] * ws.x[n] * (l - lb);
                    }
                    else if (lb < QUANT) {
                        dT[lb] += ws.dHx[n] * ws.x[n];
                    }
                }
            }
            for (auto &t: dT) {
                t /= ss.size();
            }
            return loss / ss.size();
        }

        double eta;
        double lambda;
    public:
        EM (double eta_ = 0.0001, double lambda_ = 1.0): eta(eta_), lambda(lambda_) {
            for (unsigned l = 0; l < QUANT; ++l) {
                double x = l  * LIMIT / QUANT;
                T[l] = -0.5 * sqr(x);
            }
            std::fill(T.begin(), T.end(), 0);
        }

        void load (string const &path) {
            ifstream is(path.c_str());
            for (auto &t: T) {
                is >> t;
            }
        }

        void save (string const &path) {
            ofstream os(path.c_str());
            for (auto const &t: T) {
                os << t << std::endl;
            }
        }

        static void load (string const &path, vector<Sample> *samples) {
            ifstream is(path.c_str());
            BOOST_VERIFY(is);
            Sample s;
            samples->clear();
            while (is >> s.sid >> s.target >> s.mu >> s.sigma) {
                samples->push_back(s);
            }
        }

        double train (vector<Sample> const &ss) {
            double loss = backward(ss);
            for (unsigned i = 0; i < QUANT; ++i){
                T[i] *= lambda;
                T[i] -= eta * dT[i];
            }
            return loss;
        }

        double predict (vector<Sample> *ss) {
            double loss = 0;
            for (auto &s: *ss) {
                loss += forward(&s);
            }
            return loss / ss->size();
        }
    };
}

