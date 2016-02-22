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
        static unsigned constexpr D = 17;
        struct Sample {
            string sid;
            double target;      // real value
            array<double, D> v; // features
            vector<double> cdf; // prediction for submit

            /*
            void gaussianAcc () {
                cdf.resize(N);
                for (unsigned i = 0; i < N; ++i) {
                    double x = -(double(i) - mu) / sigma;
                    cdf[i] = 0.5*erfc(x);
                }
            }
            */
        };
    private:
        array<double, D> A, dA;  // linear parameter of mu
        array<double, D> B, dB;  // linear parameter of sigma

        struct WS {
            double mu, sigma;
            array<double, N> f;
            array<double, N> x;
            array<double, N> P;
            array<double, N> dHx;
            double X;
            double loss;
        };

        void prep (Sample const &s, WS *ws) {
            double mu = 0, sigma = 0;
            for (unsigned d = 0; d < D; ++d) {
                mu += s.v[d] * A[d];
                sigma += s.v[d] * B[d];
            }
            ws->mu = mu;
            ws->sigma = sigma;
            // x_i = exp(T[l])
            ws->X = 0;
            for (unsigned n = 0; n < N; ++n) {
                double l = (mu - n) / sigma;
                double v = exp(-0.5 * l * l);
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

        double backward (Sample const &s) {
            WS ws;
            prep(s, &ws);
            std::fill(dA.begin(), dA.end(), 0);
            std::fill(dB.begin(), dB.end(), 0);
            for (unsigned n = 0; n < N; ++n) {
                double z = (ws.mu - n) / ws.sigma;
                double ma = -ws.dHx[n] * ws.x[n] * z;
                double mb = ws.dHx[n] * ws.x[n] * z * z / ws.sigma;
                for (unsigned d = 0; d < D; ++d) {
                    dA[d] += ma * s.v[d];
                    dB[d] += mb * s.v[d];
                }
            }
            return ws.loss;
        }

        double eta;
        double lambda;
    public:
        EM (double eta_ = 0.0001, double lambda_ = 1.0): eta(eta_), lambda(lambda_) {
            std::fill(A.begin(), A.end(), 0);
            A[D-1] = 1.0;
            std::fill(B.begin(), B.end(), 0);
            B[D-1] = 0.1;
        }

        void load (string const &path) {
            ifstream is(path.c_str());
            for (unsigned d = 0; d < D; ++d) {
                is >> A[d] >> B[d];
            }
        }

        void save (string const &path) {
            ofstream os(path.c_str());
            for (unsigned d = 0; d < D; ++d) {
                os << A[d] << '\t' << B[d] << std::endl;
            }
        }

        static void load (string const &path, vector<Sample> *samples) {
            ifstream is(path.c_str());
            BOOST_VERIFY(is);
            Sample s;
            samples->clear();
            while (is >> s.sid >> s.target) {
                BOOST_VERIFY(s.sid.find('_') != string::npos);
                s.v[0] = 1.0;
                for (unsigned d = 1; d < D; ++d) {
                    is >> s.v[d];
                }
                s.v[D-2]/=1000;
                samples->push_back(s);
            }
        }

        double train (vector<Sample> const &ss) {
            double loss = 0;
            array<double, D> xA;
            array<double, D> xB;
            std::fill(xA.begin(), xA.end(), 0);
            std::fill(xB.begin(), xB.end(), 0);
            for (auto const &s: ss) {
                loss += backward(s);
                for (unsigned d = 0; d < D; ++d) {
                    xA[d] += dA[d];
                    xB[d] += dB[d];
                }
            }
            for (unsigned d = 0; d < D; ++d){
                A[d] *= lambda;
                A[d] -= eta * xA[d] / ss.size();
                B[d] *= lambda;
                B[d] -= eta * xB[d] / ss.size();
            }
            return loss / ss.size();
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

