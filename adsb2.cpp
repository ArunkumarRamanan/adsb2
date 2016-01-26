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

extern "C" {
void    openblas_set_num_threads (int);    
}

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

    void GlobalInit (char const *path, Config const &config) {
        FLAGS_minloglevel = 1;
        google::InitGoogleLogging(path);
        dicom_setup(path, config);
        //openblas_set_num_threads(config.get<int>("adsb2.threads.openblas", 1));
        cv::setNumThreads(config.get<int>("adsb2.threads.opencv", 1));
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

    // histogram equilization
    void getColorMap (Stack const &stack, vector<float> *cmap, int colors) {
        vector<uint16_t> all;
        all.reserve(stack.front().image.total() * stack.size());
        for (auto const &s: stack) {
            cv::Mat const &image = s.raw;
            CHECK(image.type() == CV_16UC1);
            for (int i = 0; i < image.rows; ++i) {
                uint16_t const *p = image.ptr<uint16_t const>(i);
                for (int j = 0; j < image.cols; ++j) {
                    all.push_back(p[j]);
                }
            }
        }
        sort(all.begin(), all.end());
        cmap->resize(all.back()+1);
        unsigned b = 0;
        for (unsigned c = 0; c < colors; ++c) {
            if (b >= all.size()) break;
            unsigned e0 = all.size() * (c + 1) / colors;
            unsigned e = e0;
            // extend e0 to color all of the same color
            while ((e < all.size()) && (all[e] == all[e0])) ++e;
            unsigned cb = all[b];
            unsigned ce = (e < all.size()) ? all[e] : (all.back() + 1);
            // this is the last color bin
            // check that we have covered all colors
            if ((c+1 >= colors) && (ce != all.back() + 1)) {
                LOG(WARNING) << "bad color mapping";
                ce = all.back() + 1;
            }
            for (unsigned i = cb; i < ce; ++i) {
                cmap->at(i) = c;
            }
            b = e;
        }
    }

    void equalize (cv::Mat from, cv::Mat *to, vector<float> const &cmap) {
        CHECK(from.type() == CV_16UC1);
        to->create(from.size(), CV_32FC1);
        for (int i = 0; i < from.rows; ++i) {
            uint16_t const *f = from.ptr<uint16_t const>(i);
            float *t = to->ptr<float>(i);
            for (int j = 0; j < from.cols; ++j) {
                t[j] = cmap[f[j]];
            }
        }
    }

    void Cook::apply (Stack *stack) const {
        // normalize color
        cv::Mat mu, sigma;
        stack->getAvgStdDev(&mu, &sigma);
        // compute var image
        cv::Mat vimage;
        cv::normalize(sigma, vimage, 0, color_bins-1, cv::NORM_MINMAX, CV_32FC1);
        float scale = -1;
        float raw_spacing = -1;
        cv::Size sz;
        if (spacing > 0) {
            //float scale = spacing / meta.spacing;
            raw_spacing = stack->front().meta.raw_spacing;
            scale = raw_spacing / spacing;
            sz = round(sigma.size() * scale);
            cv::resize(vimage, vimage, sz);
        }
        vector<float> cmap;
        getColorMap(*stack, &cmap, color_bins);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < stack->size(); ++i) {
            auto &s = stack->at(i);
            if (s.do_not_cook) continue;
            equalize(s.raw, &s.image, cmap);
            if (scale > 0) {
                s.meta.spacing = spacing;
                CHECK(s.meta.raw_spacing == raw_spacing);
                CHECK(s.image.size() == sigma.size());
                //float scale = s.meta.raw_spacing / s.meta.spacing;
                cv::resize(s.image, s.image, sz);
                if (s.annotated) {
                    s.box = s.box * scale;
                }
            }
#pragma omp critical
            s.vimage = vimage;
        }
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

    void Var2Prob (cv::Mat oin, cv::Mat *out, float pth, int mk) {
        cv::Mat in;
        pth = percentile<float>(oin, pth);
        cv::threshold(oin, in, pth, 1.0, cv::THRESH_BINARY);
        cv::Mat kernel = cv::Mat::ones(mk, mk, CV_32F);
        cv::morphologyEx(in, in, cv::MORPH_OPEN, kernel);
        cv::Mat tmp = in.mul(oin);
        in = tmp;


        cv::Mat s(in.total(), 2, CV_32F);
        {
            int o = 0;
            for (int i = 0; i < in.rows; ++i) {
                for (int j = 0; j < in.cols; ++j) {
                    float *ptr = s.ptr<float>(o++);
                    ptr[0] = i;
                    ptr[1] = j;
                }
            }
            CHECK(o == s.rows);
        }
        Gaussian g(s, in.reshape(1, s.rows));
        cv::Mat p = g.prob(s);
        *out = p.reshape(1, in.rows);
        CHECK(out->size() == in.size());
    }

}

