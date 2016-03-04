#include <random>
#include <boost/regex.hpp>
#include "adsb2.h"

namespace adsb2 { namespace xg {

    using std::default_random_engine;

    static void join (vector<string> const &v, string *acc) {
        ostringstream os;
        for (unsigned i = 0; i < v.size(); ++i) {
            if (i) os << ' ';
            os << v[i];
        }
        *acc = os.str();
    }

    void run_xgboost (fs::path const &train,
                      fs::path const &test,
                      fs::path const &model,
                      fs::path const &cmdout,
                      fs::path const &out,
                      fs::path const &err,
                      Params const &params) {
        LOG(INFO) << "RUNNING XGBOOST";
        vector<string> cmd;
        cmd.push_back("OMP_NUM_THREADS=1");
        cmd.push_back((home_dir/fs::path("xgboost")).native());
        cmd.push_back((home_dir/fs::path("xglinear.conf")).native());
        cmd.push_back(fmt::format("data={}", train.native()));
        if (!test.empty()) {
            cmd.push_back(fmt::format("eval[test]={}", test.native()));
        }
        if (model.empty()) {
            cmd.push_back("model_out=/dev/null");
        }
        else {
            cmd.push_back(fmt::format("model_out={}", model.native()));
        }
        cmd.push_back(fmt::format("num_round={}", params.round));
        if (out.empty()){
            cmd.push_back(">/dev/null");
        }
        else {
            cmd.push_back(fmt::format("> {}", out.native()));
        }
        if (!err.empty()) {
            cmd.push_back(fmt::format("2> {}", err.native()));
        }
        string x;
        join(cmd, &x);
        if (!cmdout.empty()) {
            fs::ofstream os(cmdout);
            os << x << std::endl;
        }
        ::system(x.c_str());
    }

    void probe (fs::path const &train,
                  fs::path const &test,
                  Params const &params,
                  float *log) {
        fs::path log_path = temp_path();
        run_xgboost(train, test, fs::path(), fs::path(), log_path, fs::path(), params);
        fs::ifstream is(log_path);
        string line;
        // [21:27:58] [2]  test-rmse:18.968014
        boost::regex regex("^\\[.+\\]\\s+\\[([0-9]+)\\]\\s+test-rmse:([0-9.]+)$");
        int n = 0;
        while (getline(is, line)) {
            boost::smatch what;
            if (boost::regex_match(line, what, regex, boost::match_extra)) {
                CHECK(lexical_cast<int>(what[1]) == n);
                log[n] = lexical_cast<float>(what[2]);
                ++n;
            }
        }
        CHECK(n == params.round);
        fs::remove(log_path);
    }

    void train (fs::path const &train,
                  fs::path const &model,
                  fs::path const &test,
                  Params const &params) {
    }



    void XGProbe (fs::path const &train,
                  fs::path const &test,
                  int round,
                  float *log) {
    }

    static void bootstrap (vector<string> const &lines,
            fs::path const &train_path,
            fs::path const &test_path,
            default_random_engine &e) {
        vector<bool> mask(lines.size(), false);
        {
            fs::ofstream os(train_path);
            for (unsigned i = 0; i < lines.size(); ++i) {
                unsigned idx = e() % lines.size();
                os << lines[idx] << std::endl;
                mask[idx] = true;
            }
        }
        {
            fs::ofstream os(test_path);
            for (unsigned i = 0; i < lines.size(); ++i)
                if (!mask[i]) {
                    os << lines[i] << std::endl;
                }
        }
    }

    struct OptRange {
        int opt;
        int begin, end;
        float rmse, max;
    };
    bool operator < (OptRange const &r1, OptRange const &r2) {
        return r1.begin < r2.begin;
    }

    void tune (fs::path const &path, TuneParams const &tp, TuneResult *result) {
        vector<string> lines;
        {
            fs::ifstream is(path);
            string line;
            while (getline(is, line)) {
                lines.emplace_back(std::move(line));
            }
        }
        default_random_engine rng(tp.seed);
        fs::path temp_train = temp_path();
        fs::path temp_test = temp_path();
        Params pp;
        pp.round = tp.max_round;
        cv::Mat log(tp.max_it, tp.max_round, CV_32F);
        vector<OptRange> opts;
        vector<float> sum(tp.max_round, 0);
        for (int it = 0; it < tp.max_it; ++it) {
            bootstrap(lines, temp_train, temp_test, rng);
            float *ptr = log.ptr<float>(it);
            probe(temp_train, temp_test, pp, ptr);
            OptRange range;
            range.opt = std::min_element(ptr, ptr + tp.max_round) - ptr;
            range.rmse = ptr[range.opt];
            range.max = ptr[tp.max_round-1];
            range.begin = range.opt;
            float ub = range.rmse + tp.tolerate;
            while ((range.begin > 0) && (ptr[range.begin-1] <= ub)) --range.begin;
            range.end = range.opt+1;
            while ((range.end < tp.max_round) && (ptr[range.end] <= ub)) ++range.end;
            opts.push_back(range);
            for (unsigned i = 0; i < tp.max_round; ++i) {
                sum[i] += ptr[i];
            }
        }
        fs::remove(temp_train);
        fs::remove(temp_test);
        sort(opts.begin(), opts.end());
        for (auto const &p: opts) {
            std::cout << p.begin << '\t' << p.opt << '\t' << p.end << '\t' << p.rmse << '\t' << p.max << std::endl;
        }
        vector<int> vote(tp.max_round, 0);
        for (auto const &p: opts) {
            for (int i = p.begin; i < p.end; ++i) {
                ++vote[i];
            }
        }
        int lb = 0;
        int ub = 0;
        for (int i = 1; i < vote.size(); ++i) {
            if (vote[i] > vote[lb]) {
                lb = i;
            }
            if (vote[i] >= vote[ub]) {
                ub = i;
            }
        }
        result->round1 = (lb + ub)/2;
        result->round2 = std::min_element(sum.begin(), sum.end()) - sum.begin();
        std::cout << lb << '\t' << ub << '\t' << result->round1 << '\t' << result->round2 << std::endl;
    }

}}
