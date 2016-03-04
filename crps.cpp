#include <boost/program_options.hpp>
#include "adsb2.h"

using namespace std;
using namespace adsb2;

int main (int argc, char *argv[]) {
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;

    float mu, sigma;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("mu,m", po::value(&mu)->default_value(10), "")
    ("sigma,s", po::value(&sigma)->default_value(5), "")
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
    GaussianAcc acc(config);
    vector<float> v;
    acc.apply(mu,sigma, &v);
    for (unsigned i = 0; i < Eval::VALUES; ++i) {
        cout << i << '\t' << v[i] << endl;
    }
    
}
