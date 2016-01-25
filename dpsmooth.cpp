#include <cmath>
#include <random>
#include <iostream>
#include <boost/multi_array.hpp>
#include <boost/timer/timer.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/program_options.hpp>

using namespace std;

struct Cell {
    float cost;
    unsigned ptr;
};

struct Task {
    vector<float> data;
    void optimize (unsigned M) {
        auto minmax = minmax_element(data.begin(), data.end());
        float vmin = *minmax.first;
        float vmax = *minmax.second;
        BOOST_VERIFY(vmin <= vmax);
        float step = (vmax - vmin) / (M - 1);
        unsigned N = data.size();
        boost::multi_array<Cell, 2> table(boost::extents[N][M]);
        // initialize for n == 0
        {
            Cell *row = &table[0][0];
            float y = data[0];
            for (unsigned m = 0; m < M; ++m) {
                float yp = vmin + m * step; 
                row[m].cost = fabs(y - yp);
                row[m].ptr = 0; // initialize anyway
            }
        }
        for (unsigned i = 1; i < data.size(); ++i) {
            Cell const *row0 = &table[i-1][0];
            Cell *row = &table[i][0];
            float y = data[i];
            unsigned best = 0;
            for (unsigned m = 0; m < M; ++m) {
                if (row0[m].cost < row0[best].cost) {
                    best = m;
                }
                float yp = vmin + m * step; 
                row[m].cost = fabs(yp - y) + row0[best].cost;
                row[m].ptr = best;
            }
        }
        // find best solution
        {
            unsigned off = data.size() - 1;
            vector<unsigned> path; // initially reversed
            unsigned best = 0;
            {   // last step, find smallest value
                Cell *row = &table[off][0];
                for (unsigned m = 1; m < M; ++m) {
                    if (row[m].cost < row[best].cost) {
                        best = m;
                    }
                }
                path.push_back(best);
            }
            while (off > 0) {
                best = table[off][best].ptr;
                path.push_back(best);
                --off;
            }
            BOOST_VERIFY(path.size() == data.size());
            reverse(path.begin(), path.end());
            for (unsigned i = 0; i < path.size(); ++i){
                data[i] = vmin + path[i] * step;
            }
        }
    }
};

int main (int argc, char *argv[]) {
    namespace po = boost::program_options; 
    unsigned N, M, R = 0;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    (",N", po::value(&N)->default_value(600), "")
    (",M", po::value(&M)->default_value(1000),"")
    ("random", po::value(&R), "generate random testing data without optimization")
    ;

    po::positional_options_description p;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help")) {
        cerr << desc;
        return 1;
    }

    vector<Task> tasks;

    if (vm.count("random")) {
        // generate random data for testing
        cerr << "Generating random data..." << endl;
        tasks.resize(R);
        default_random_engine gen;
        uniform_real_distribution<float> u(0, 1);
        for (auto &task: tasks) {
            auto &data = task.data;
            data.resize(N);
            for (auto &a: data) {
                a = u(gen);
            }
        }
    }
    else {   // read input
        vector<float> v;
        for (;;) {
            v.resize(N);
            for (auto &a: v) {
                cin >> a;
            }
            if (!cin) break;
            tasks.push_back(Task());
            tasks.back().data.swap(v);
        }
        cerr << tasks.size() << " lines read." << endl;
        {
            boost::timer::auto_cpu_timer timer(cerr);
            cerr << "Optimizing..." << endl;
            boost::progress_display progress(tasks.size(), cerr);
#pragma omp parallel for schedule(dynamic, 1)
            for (unsigned i = 0; i < tasks.size(); ++i) {
                tasks[i].optimize(M);
#pragma omp critical
                ++progress;
            }
        }
    }
    for (auto const &task: tasks) {
        auto const &v = task.data;
        for (unsigned i = 0;i < v.size(); ++i) {
            if (i) cout << '\t';
            cout << v[i];
        }
        cout << endl;
    }


    return 0;
}
