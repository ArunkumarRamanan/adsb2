#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "adsb2.h"

using namespace std;
using namespace cv;
using namespace adsb2;

extern char const *HEADER;

struct Case {
    int id;
    Volume min;
    Volume max;
};

void GaussianAcc (Volume const &v, float scale, vector<float> *s) {
    s->clear();
    for (unsigned i = 0; i < Eval::VALUES; ++i) {
        float x = -(float(i) * 1000 - v.mean) / scale;
        s->push_back(0.5*erfc(x));
    }
}

int main(int argc, char **argv) {
    //Series stack("sax", "tmp");
    namespace po = boost::program_options; 
    string config_path;
    vector<string> overrides;
    vector<string> paths;
    float scale;
    bool do_eval = false;
#define DO_LINEAR 1
#ifdef DO_LINEAR
    float sa, sb;
    float da, db;
#endif

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path)->default_value("adsb2.xml"), "config file")
    ("override,D", po::value(&overrides), "override configuration.")
    ("input,i", po::value(&paths), "")
    ("scale,s", po::value(&scale)->default_value(80), "")
#ifdef DO_LINEAR
    ("sa", po::value(&sa)->default_value(1.0), "")
    ("sb", po::value(&sb)->default_value(0.0), "")
    ("da", po::value(&da)->default_value(1.0), "")
    ("db", po::value(&db)->default_value(0.0), "")
#endif
    ("eval", "")
    ;


    po::positional_options_description p;
    p.add("input", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || paths.empty()) {
        cerr << desc;
        return 1;
    }
    if (vm.count("eval")) do_eval = true;
    vector<Case> cases;
    for (auto const &p: paths) {
        fs::path path(p);
        Case c;
        c.id = lexical_cast<int>(path.parent_path().filename().native());
        fs::ifstream is(path);
        CHECK(is);
        is >> c.min.mean >> c.min.var >> c.max.mean >> c.max.var;
#ifdef DO_LINEAR
        c.min.mean *= sa;
        c.min.mean += sb;
        c.max.mean *= da;
        c.max.mean += db;
#endif
        c.min.var *= c.min.var;
        c.max.var *= c.max.var;
        CHECK(is);
        cases.push_back(c);
    }
    sort(cases.begin(), cases.end(),
            [](Case const &c1, Case const &c2) {
                return c1.id < c2.id;
            });
    if (!do_eval) {
        cout << HEADER << endl;
    }
    Eval eval;
    for (auto const &c: cases) {
        vector<float> v;
        GaussianAcc(c.max, scale, &v);
        cout << c.id << "_Diastole";
        if (do_eval) {
            float s = eval.score(c.id, 1, v);
            cout << '\t' << s << '\t' <<  (eval.get(c.id, 1) - c.max.mean/1000) << '\t' << eval.get(c.id, 1) << '\t' << c.max.mean/1000 << '\t' << sqrt(c.max.var)/1000;
        }
        else {
            for (auto const &f: v) {
                cout << ',' << f;
            }
        }
        cout << endl;
        GaussianAcc(c.min, scale, &v);
        //
        cout << c.id << "_Systole";
        if (do_eval) {
            float s = eval.score(c.id, 0, v);
            cout << '\t' << s << '\t' << (eval.get(c.id, 0) - c.min.mean/1000) << '\t' << eval.get(c.id, 0) << '\t' << c.min.mean/1000 << '\t' << sqrt(c.min.var)/1000;
        }
        else {
            for (auto const &f: v) {
                cout << ',' << f;
            }
        }
        cout << endl;
    }
}

char const *HEADER = "Id,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19,P20,P21,P22,P23,P24,P25,P26,P27,P28,P29,P30,P31,P32,P33,P34,P35,P36,P37,P38,P39,P40,P41,P42,P43,P44,P45,P46,P47,P48,P49,P50,P51,P52,P53,P54,P55,P56,P57,P58,P59,P60,P61,P62,P63,P64,P65,P66,P67,P68,P69,P70,P71,P72,P73,P74,P75,P76,P77,P78,P79,P80,P81,P82,P83,P84,P85,P86,P87,P88,P89,P90,P91,P92,P93,P94,P95,P96,P97,P98,P99,P100,P101,P102,P103,P104,P105,P106,P107,P108,P109,P110,P111,P112,P113,P114,P115,P116,P117,P118,P119,P120,P121,P122,P123,P124,P125,P126,P127,P128,P129,P130,P131,P132,P133,P134,P135,P136,P137,P138,P139,P140,P141,P142,P143,P144,P145,P146,P147,P148,P149,P150,P151,P152,P153,P154,P155,P156,P157,P158,P159,P160,P161,P162,P163,P164,P165,P166,P167,P168,P169,P170,P171,P172,P173,P174,P175,P176,P177,P178,P179,P180,P181,P182,P183,P184,P185,P186,P187,P188,P189,P190,P191,P192,P193,P194,P195,P196,P197,P198,P199,P200,P201,P202,P203,P204,P205,P206,P207,P208,P209,P210,P211,P212,P213,P214,P215,P216,P217,P218,P219,P220,P221,P222,P223,P224,P225,P226,P227,P228,P229,P230,P231,P232,P233,P234,P235,P236,P237,P238,P239,P240,P241,P242,P243,P244,P245,P246,P247,P248,P249,P250,P251,P252,P253,P254,P255,P256,P257,P258,P259,P260,P261,P262,P263,P264,P265,P266,P267,P268,P269,P270,P271,P272,P273,P274,P275,P276,P277,P278,P279,P280,P281,P282,P283,P284,P285,P286,P287,P288,P289,P290,P291,P292,P293,P294,P295,P296,P297,P298,P299,P300,P301,P302,P303,P304,P305,P306,P307,P308,P309,P310,P311,P312,P313,P314,P315,P316,P317,P318,P319,P320,P321,P322,P323,P324,P325,P326,P327,P328,P329,P330,P331,P332,P333,P334,P335,P336,P337,P338,P339,P340,P341,P342,P343,P344,P345,P346,P347,P348,P349,P350,P351,P352,P353,P354,P355,P356,P357,P358,P359,P360,P361,P362,P363,P364,P365,P366,P367,P368,P369,P370,P371,P372,P373,P374,P375,P376,P377,P378,P379,P380,P381,P382,P383,P384,P385,P386,P387,P388,P389,P390,P391,P392,P393,P394,P395,P396,P397,P398,P399,P400,P401,P402,P403,P404,P405,P406,P407,P408,P409,P410,P411,P412,P413,P414,P415,P416,P417,P418,P419,P420,P421,P422,P423,P424,P425,P426,P427,P428,P429,P430,P431,P432,P433,P434,P435,P436,P437,P438,P439,P440,P441,P442,P443,P444,P445,P446,P447,P448,P449,P450,P451,P452,P453,P454,P455,P456,P457,P458,P459,P460,P461,P462,P463,P464,P465,P466,P467,P468,P469,P470,P471,P472,P473,P474,P475,P476,P477,P478,P479,P480,P481,P482,P483,P484,P485,P486,P487,P488,P489,P490,P491,P492,P493,P494,P495,P496,P497,P498,P499,P500,P501,P502,P503,P504,P505,P506,P507,P508,P509,P510,P511,P512,P513,P514,P515,P516,P517,P518,P519,P520,P521,P522,P523,P524,P525,P526,P527,P528,P529,P530,P531,P532,P533,P534,P535,P536,P537,P538,P539,P540,P541,P542,P543,P544,P545,P546,P547,P548,P549,P550,P551,P552,P553,P554,P555,P556,P557,P558,P559,P560,P561,P562,P563,P564,P565,P566,P567,P568,P569,P570,P571,P572,P573,P574,P575,P576,P577,P578,P579,P580,P581,P582,P583,P584,P585,P586,P587,P588,P589,P590,P591,P592,P593,P594,P595,P596,P597,P598,P599";
