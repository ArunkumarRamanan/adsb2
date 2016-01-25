#include <unordered_map>
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/xml_parser.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include "adsb2.h"

namespace adsb2 {
    using std::unordered_map;
    void LoadConfig (string const &path, Config *config) {
        try {
            boost::property_tree::read_xml(path, *config);
        }
        catch (...) {
            LOG(WARNING) << "Cannot load config file " << path << ", using defaults.";
        }
    }

    void SaveConfig (string const &path, Config const &config) {
        boost::property_tree::write_xml(path, config);
    }

    void OverrideConfig (std::vector<std::string> const &overrides, Config *config) {
        for (std::string const &D: overrides) {
            size_t o = D.find("=");
            if (o == D.npos || o == 0 || o + 1 >= D.size()) {
                std::cerr << "Bad parameter: " << D << std::endl;
                BOOST_VERIFY(0);
            }
            config->put<std::string>(D.substr(0, o), D.substr(o + 1));
        }
    }

    Sample::Sample (string const &txt): do_not_cook(false) {
        istringstream ss(txt);
        string p;
        ss >> p >> box.x >> box.y >> box.width >> box.height;
        if (!ss) {
            box.x = box.y = -1;
            box.width = box.height = 0;
            return;
        }
        annotated = true;
        path = fs::path(p);
        line = txt;
    }

    void Sample::eval (cv::Mat mat, float *s1, float *s2) const {
        CHECK(box.x >=0 && box.y >= 0);
        cv::Mat roi = mat(round(box));
        float total = cv::sum(mat)[0];
        float covered = cv::sum(roi)[0];
        *s1 = covered / total;
        *s2 = 0;//*s2 = std::sqrt(roi.area()) * meta.spacing;
    }

    Stack::Stack (fs::path const &input_dir, bool load) {
        // enumerate DCM files
        vector<fs::path> paths;
        fs::directory_iterator end_itr;
        for (fs::directory_iterator itr(input_dir);
                itr != end_itr; ++itr) {
            if (fs::is_regular_file(itr->status())) {
                // found subdirectory,
                // create tagger
                auto path = itr->path();
                auto ext = path.extension();
                if (ext.string() != ".dcm") {
                    LOG(WARNING) << "Unknown file type: " << path.string();
                    continue;
                }
                paths.push_back(path);
            }
        }
        std::sort(paths.begin(), paths.end());
        resize(paths.size());
        for (unsigned i = 0; i < paths.size(); ++i) {
            Sample &s = at(i);
            s.path = paths[i];
            if (load) {
                s.load_raw();
                if (i) {
                    CHECK(s.meta.spacing == at(0).meta.spacing);
                    CHECK(s.image.size() == at(0).image.size());
                }
            }
            /*
            auto const &name = names[i];
            auto dcm_path = input_dir;
            dcm_path /= name;
            dcm_path += ".dcm";
            cv::Mat image = loader.load(dcm_path.native(), &metas[i]);
            //ImageAdaptor::apply(&image);
            BOOST_VERIFY(image.total());
            BOOST_VERIFY(image.type() == CV_16U);
            BOOST_VERIFY(image.isContinuous());
            at(i) = image;
            */
        }
    }

    void Stack::getAvgStdDev (cv::Mat *avg, cv::Mat *stddev) {
        namespace ba = boost::accumulators;
        typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::count, ba::tag::variance, ba::tag::moment<2>>> Acc;
        CHECK(size());
        CHECK(at(0).image.type() == CV_16U);

        cv::Size shape = at(0).image.size();
        unsigned pixels = shape.area();

        cv::Mat mu(shape, CV_32F);
        cv::Mat sigma(shape, CV_32F);
        vector<Acc> accs(pixels);
        for (auto const &s: *this) {
            uint16_t const *v = s.image.ptr<uint16_t const>(0);
            for (auto &acc: accs) {
                acc(*v);
                ++v;
            }
        }
        float *m = mu.ptr<float>(0);
        float *s = sigma.ptr<float>(0);
        //float *sp = spread.ptr<float>(0);
        for (auto const &acc: accs) {
            *m = ba::mean(acc);
            *s = std::sqrt(ba::variance(acc));
            //cout << *s << endl;
            //*sp = ba::max(acc) - ba::min(acc);
            ++m; ++s; //++sp;
        }
        *avg = mu;
        *stddev = sigma;
    }

    void shrink_expand (vector<uint16_t> &all, vector<uint16_t> *v, float eth) {
        percentile(all, vector<float>{0, eth, 1 - eth, 1}, v);
        CHECK(v->size() == 4);
        int r = v->at(2) - v->at(1);
        int low = v->at(1) - r * eth;
        int high = v->at(2) + r * eth;
        if (low < v->at(0)) low = v->at(0);
        if (high > v->at(3)) high = v->at(3);
        v->resize(2);
        v->at(0) = low;
        v->at(1) = high;
        CHECK(v->size() == 2);
    }

    // get color range from RAW images of the stack
    // sigma is precomputed standard deviation image
    void getColorRange (Stack const &stack, cv::Mat sigma, ColorRange *range, float vth, float eth) {
        vector<uint16_t> all;
        all.reserve(stack.front().image.total() * stack.size());
        vector<uint16_t> roi;
        vector<vector<int>> picked;
        if (stack.size() > 1) { // has sigma
            roi.reserve(stack.front().image.total() * stack.size());
            float sth = percentile<float>(sigma, vth);  // sigma th
            picked.resize(sigma.rows);
            for (int i = 0; i < sigma.rows; ++i) {
                auto &v = picked[i];
                float const *p = sigma.ptr<float const>(i);
                for (int j = 0; j < sigma.cols; ++j) {
                    if (p[j] >= vth) {
                        v.push_back(j);
                    }
                }
            }
        }

        for (auto const &s: stack) {
            cv::Mat const &image = s.raw;
            for (int i = 0; i < image.rows; ++i) {
                auto const &v = picked[i];
                uint16_t const *p = image.ptr<uint16_t const>(i);
                for (int j = 0; j < image.cols; ++j) {
                    all.push_back(p[j]);
                }
                for (int j: v) {
                    roi.push_back(p[j]);
                }
            }
        }
        vector<uint16_t> allth;
        vector<uint16_t> roith;
        shrink_expand(all, &allth, eth);
        shrink_expand(roi, &roith, eth);
        range->min = allth[0];
        range->max = allth[1];
        range->umin = roith[0];
        range->umax = roith[1];
        if (range->umin < range->min) {
            LOG(WARNING) << "exand min " << range->umin << " => " << range->min;
            range->umin = range->min;
        }
        if (range->umax > range->max) {
            LOG(WARNING) << "exand max " << range->umax << " => " << range->max;
            range->umax = range->max;
        }
    }

    void scaleColor (cv::Mat from, cv::Mat *to, ColorRange const &range,
                     float tlow, float thigh, float tmax) {
        CHECK(from.type() == CV_16UC1);
        to->create(from.size(), CV_32FC1);
        // [low, ulow) -> [0, tlow)
        // [ulow, uhigh] -> [tlow, thigh]
        // (uhigh, high] -> (thigh, 255]
        int low = range.min;
        int ulow = range.umin;
        int high = range.max;
        int uhigh = range.umax;


        if (ulow - low < tlow) {
            // e.g.   min = 100, low = 101
            //                   tlow = 10
            // we'll lower min = 101-10 = 91, so there's
            // 1-1 correspondance between [91, 101) and [0, 10)
            low = ulow - tlow;
        }
        if (high - uhigh < tmax - thigh) {
            // e.g.   high = 100, max = 101
            //        thigh = 245,
            // we raise max = 100 + 255 - 245 = 110
            //        [100, 110] <-> [245, 255]
            high = uhigh + (tmax - thigh);

        }
        for (int i = 0; i < from.rows; ++i) {
            uint16_t const *f = from.ptr<uint16_t const>(i);
            float *t = to->ptr<float>(i);
            for (int j = 0; j < from.cols; ++j) {
                uint16_t x = f[j];
                float y = 0;
                if (x < low) {
                    y = 0;
                }
                if (x < ulow) {
                    // [low, ulow) -> [0, tlow)
                    y = (x - low) * tlow / (ulow - low);
                }
                else if (x <= uhigh) {
                    // [ulow, uhigh] -> [tlow, thigh]
                    y = (x - ulow) * (thigh - tlow) / (uhigh - ulow) + tlow;
                    // x == ulow  ==> tlow
                    // x == uhigh ==> thigh
                }
                else if (x <= high) {
                    // (uhigh, high] -> (thigh, 255]
                     y = (x - uhigh) * (tmax - thigh) / (high - uhigh) + thigh;
                    // uhigh =>  thigh
                    // high => 255
                }
                else {
                    y = tmax;
                }
                if (!(y > 0)) {
                    if (!(y > -1)) {
                        LOG(WARNING) << "y < 0: " << y;
                    }
                    y = 0;
                }
                if (!(y <= tmax)) {
                    LOG(WARNING) << "y > tmax: " << y << " " << tmax;
                    y = tmax;
                }
                t[j] = y;
            }
        }
    }

    void Cook::apply (Stack *stack) const {
        // normalize color
        cv::Mat mu, sigma;
        stack->getAvgStdDev(&mu, &sigma);
        ColorRange cr;
        getColorRange(*stack, sigma, &cr, color_vth, color_eth);
        float clow = color_max * color_margin;
        float chigh = color_max - color_max * color_margin;
        for (auto &s: *stack) {
            if (s.do_not_cook) continue;
            scaleColor(s.raw, &s.image, cr, clow, chigh, color_max);
        }
        // compute var image
        cv::Mat vimage;
        cv::normalize(sigma, vimage, 0, color_max, cv::NORM_MINMAX, CV_32FC1);
        // normalize size
        if (spacing > 0) {
            //float scale = spacing / meta.spacing;
            float raw_spacing = stack->front().meta.raw_spacing;
            float scale = raw_spacing / spacing;
            cv::Size sz = round(stack->front().image.size() * scale);
            cv::resize(vimage, vimage, sz);
            for (auto &s: *stack) {
                if (s.do_not_cook) continue;
                s.meta.spacing = spacing;
                CHECK(s.meta.raw_spacing == raw_spacing);
                //float scale = s.meta.raw_spacing / s.meta.spacing;
                cv::resize(s.image, s.image, round(s.image.size() * scale));
                if (s.annotated) {
                    s.box = s.box * scale;
                }
            }
        }
        for (auto &s: *stack) {
            if (s.do_not_cook) continue;
            s.vimage = vimage;
        }
        // compute label image
    }

    Samples::Samples (fs::path const &list_path, fs::path const &root, Cook const &cook) {
        fs::ifstream is(list_path);
        CHECK(is) << "Cannot open list file: " << list_path;
        string line;
        while (getline(is, line)) {
            Sample s(line);
            if (s.line.empty()) {
                LOG(ERROR) << "bad line: " << line;
                continue;
            }
            if (s.path.extension().string() != ".dcm") {
                LOG(ERROR) << "not DCM file: " << s.path;
                continue;
            }
            fs::path f = root;
            f /= s.path;
            if (!fs::is_regular_file(f)) {
                LOG(ERROR) << "not regular file: " << f;
                continue;
            }
            push_back(s);
        }
        // distribute samples to stacks
        std::unordered_map<string, vector<unsigned>> dirs;
        for (unsigned i = 0; i < size(); ++i) {
            dirs[at(i).path.parent_path().string()].push_back(i);
        }
        LOG(INFO) << "found files in " << dirs.size() << " dirs.";
    
        boost::progress_display progress(dirs.size(), std::cerr);
        vector<std::pair<fs::path, vector<unsigned>>> todo;
        for (auto &p: dirs) {
            todo.emplace_back(fs::path(p.first), std::move(p.second));
        }
#pragma omp parallel for
        for (unsigned ii = 0; ii < todo.size(); ++ii) {
            fs::path dir = root / fs::path(todo[ii].first);
            Stack stack(dir);
            vector<std::pair<unsigned, unsigned>> offs;
            {
                unordered_map<string, unsigned> mm;
                for (unsigned i = 0; i < stack.size(); ++i) {
                    mm[stack[i].path.stem().string()] = i;
                }
                // offset mapping: from samples offset to stack offset
                for (unsigned i: todo[ii].second) {
                    auto it = mm.find(at(i).path.stem().string());
                    CHECK(it != mm.end()) << "cannot find " << at(i).path << " in dir " << dir;
                    offs.emplace_back(i, it->second);
                }
            }
            for (auto &s: stack) {
                s.do_not_cook = true;
            }
            for (auto const &p: offs) {
                Sample &from = at(p.first);
                Sample &to = stack[p.second];
                to.do_not_cook = false;
                to.line = from.line;
                to.annotated = from.annotated;
                to.box = from.box;
            }
            // move annotation data to stack
            cook.apply(&stack);
            // now extract the files we want
            for (auto const &p: offs) {
                std::swap(at(p.first), stack[p.second]);
            }
            ++progress;
        }
    }
}

