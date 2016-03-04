#include <boost/program_options.hpp>
#include "adsb2.h"

using namespace std;
using namespace adsb2;

int main (int argc, char *argv[]) {
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;

    xg::Params params;
    //params.round = 4000;
    xg::TuneParams tp;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("fold,F", po::value(&tp.max_it)->default_value(10), "")
    ("round,R", po::value(&tp.max_round)->default_value(2500), "")
    ("tolerate,T", po::value(&tp.tolerate)->default_value(0.1), "")
    ("seed,S", po::value(&tp.seed)->default_value(2016), "")
    ;

    po::positional_options_description p;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help")) {
        cerr << "ADSB2 VERSION: " << VERSION << endl;
        cerr << desc;
        return 1;
    }

    Config config;
    try {
        LoadConfig(config_path, &config);
    } catch (...) {
        cerr << "Failed to load config file: " << config_path << ", using defaults." << endl;
    }
    OverrideConfig(overrides, &config);

    GlobalInit(argv[0], config);
    params.round = tp.max_round;
    //xg::probe("train", "test", params, nullptr);
    xg::TuneResult tuned;
    xg::tune("train", tp, &tuned);
    vector<float> log(params.round);
    xg::probe("train", "test", params, &log[0]);
    auto it = std::min_element(log.begin(), log.end());
    cout << "TEST OPT: " << it - log.begin() << '\t' << *it << endl;
    cout << "TEST PRED1: " << tuned.round1 << '\t' << log[tuned.round1] << endl;
    cout << "TEST PRED2: " << tuned.round2 << '\t' << log[tuned.round2] << endl;
    cout << "TEST END: " << log.back() << endl;
    
}
