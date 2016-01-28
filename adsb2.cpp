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
    char const *MetaBase::FIELDS[] = {
        "Sex",
        "Age",
        "SliceThickness",
        "NominalInterval",
        "CardiacNumberOfImages",
        "SliceLocation",
        "SeriesNumber"
    };

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
        FLAGS_logtostderr = 1;
        FLAGS_minloglevel = 1;
        google::InitGoogleLogging(path);
        dicom_setup(path, config);
        //openblas_set_num_threads(config.get<int>("adsb2.threads.openblas", 1));
        cv::setNumThreads(config.get<int>("adsb2.threads.opencv", 1));
    }

    Slice::Slice (string const &txt): do_not_cook(false) {
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

    void Slice::eval (cv::Mat mat, float *s1, float *s2) const {
        CHECK(box.x >=0 && box.y >= 0);
        cv::Mat roi = mat(round(box));
        float total = cv::sum(mat)[0];
        float covered = cv::sum(roi)[0];
        *s1 = covered / total;
        *s2 = 0;//*s2 = std::sqrt(roi.area()) * meta.spacing;
    }

    Series::Series (fs::path const &input_dir, bool load, bool check, bool fix): series_path(input_dir) {
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
        CHECK(paths.size());
        std::sort(paths.begin(), paths.end());
        resize(paths.size());
        for (unsigned i = 0; i < paths.size(); ++i) {
            Slice &s = at(i);
            s.path = paths[i];
            if (load) {
                s.load_raw();
                if (i) {
                    CHECK(s.meta.spacing == at(0).meta.spacing);
                    CHECK(s.image.size() == at(0).image.size());
                }
            }
        }
        if (load && check && !sanity_check(fix) && fix) {
            CHECK(sanity_check(false));
        }
    }

    template <typename T>
    class FreqCount {
        unordered_map<T, unsigned> cnt;
    public:
        void update (T const &v) {
            cnt[v] += 1;
        }
        bool unique () const {
            return cnt.size() <= 1;
        }
        T most_frequent () const {
            auto it = std::max_element(cnt.begin(), cnt.end(),
                    [](std::pair<T, unsigned> const &p1,
                       std::pair<T, unsigned> const &p2) {
                        return p1.second < p2.second;
                    });
            float mfv = it->first;
        }
    };

    bool operator < (Slice const &s1, Slice const &s2)
    {
        return s1.meta.trigger_time < s2.meta.trigger_time;
    }

    bool Series::sanity_check (bool fix) {
        bool ok = true;
        for (Slice &s: *this) {
            if (s.meta[Meta::NUMBER_OF_IMAGES] != size()) {
                ok = false;
                LOG(WARNING) << "Series field #images mismatch: " << s.path << " found " << s.meta[Meta::NUMBER_OF_IMAGES] << " instead of actually # images found " << size();
                if (fix) {
                    s.meta[Meta::NUMBER_OF_IMAGES] = size();
                }
            }
        }
        for (unsigned i = 0; i < Meta::SERIES_FIELDS; ++i) {
            // check that all series fields are the same
            FreqCount<float> fc;
            for (Slice &s: *this) {
                fc.update(s.meta[i]);
            }
            if (fc.unique()) break;
            ok = false;
            float mfv = fc.most_frequent();
            for (Slice &s: *this) {
                if (s.meta[i] != mfv) {
                    LOG(WARNING) << "Series field " << Meta::FIELDS[i] << "  mismatch: " << s.path << " found " << s.meta[i]
                                 << " instead of most freq value " << mfv;
                    if (fix) {
                        s.meta[i] = mfv;
                    }
                }
            }
        }
        // check trigger time
        bool ooo = false;
        for (unsigned i = 1; i < size(); ++i) {
            if (!(at(i).meta.trigger_time > at(i-1).meta.trigger_time)) {
                ooo = true;
                ok = false;
                LOG(WARNING) << "Trigger time out of order: "
                             << at(i-1).path << ':' << at(i-1).meta.trigger_time
                             << " > "
                             << at(i).path << ':' << at(i).meta.trigger_time;
            }
        }
        if (fix) {
            sort(begin(), end());
        }
        return ok;
    }

    void Series::getAvgStdDev (cv::Mat *avg, cv::Mat *stddev) {
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
    void getColorMap (Series const &stack, vector<float> *cmap, int colors) {
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

    void Cook::apply (Series *stack) const {
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

    void Cook::apply (Series *study) const {
        for (auto &s: *study) {
            apply(&s);
        }
    }

    Slices::Slices (fs::path const &list_path, fs::path const &root, Cook const &cook) {
        fs::ifstream is(list_path);
        CHECK(is) << "Cannot open list file: " << list_path;
        string line;
        while (getline(is, line)) {
            Slice s(line);
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
            Series stack(dir);
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
                Slice &from = at(p.first);
                Slice &to = stack[p.second];
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

    Study::Study (fs::path const &input_dir, bool load, bool check, bool fix): study_path(input_dir) {
        // enumerate DCM files
        vector<fs::path> paths;
        fs::directory_iterator end_itr;
        for (fs::directory_iterator itr(input_dir);
                itr != end_itr; ++itr) {
            if (fs::is_directory(itr->status())) {
                // found subdirectory,
                // create tagger
                auto path = itr->path();
                string name = path.filename().native();
                if (name.find("sax_") != 0) {
                    continue;
                }
                paths.push_back(path);
            }
        }
        std::sort(paths.begin(), paths.end());
        CHECK(paths.size());
        for (auto const &path: paths) {
            emplace_back(path, load, false, false);    // do not fix for now
        }
        if (load && !sanity_check(fix) && fix) {
            CHECK(sanity_check(false));
        }
    }

    static constexpr float LOCATION_GAP_EPSILON = 0.01;
    static inline bool operator < (Series const &s1, Series const &s2) {
        Meta const &m1 = s1.front().meta;
        Meta const &m2 = s2.front().meta;
        if (m1[Meta::SLICE_LOCATION] + LOCATION_GAP_EPSILON < m2[Meta::SLICE_LOCATION]) {
            return true;
        }
        if (m1[Meta::SLICE_LOCATION] - LOCATION_GAP_EPSILON > m2[Meta::SLICE_LOCATION]) {
            return false;
        }
        return m1[Meta::SERIES_NUMBER] < m2[Meta::SERIES_NUMBER];
    }


    bool Study::sanity_check (bool fix) {
        bool ok = true;
        if (fix) {
            check_regroup();
        }
        for (auto &s: *this) {
            if (!s.sanity_check(fix)) {
                LOG(WARNING) << "Study " << study_path << " series " << s.series_path << " sanity check failed.";
                if (fix) {
                    CHECK(s.sanity_check(false));
                }
            }
        }
        for (unsigned i = 0; i < Meta::STUDY_FIELDS; ++i) {
            // check that all series fields are the same
            FreqCount<float> fc;
            for (Series &s: *this) {
                fc.update(s.front().meta[i]);
            }
            if (fc.unique()) break;
            ok = false;
            float mfv = fc.most_frequent();
            for (Series &s: *this) {
                if (s.front().meta[i] != mfv) {
                    LOG(WARNING) << "Study field " << Meta::FIELDS[i] << "  mismatch: " << s.dir() << " found " << s.front().meta[i]
                                 << " instead of most freq value " << mfv;
                    if (fix) {
                        for (auto &ss: s) {
                            ss.meta[i] = mfv;
                        }
                    }
                }
            }
        }
        // check trigger time
        //
        sort(begin(), end());
        unsigned off = 1;
        for (unsigned i = 1; i < size(); ++i) {
            Meta const &prev = at(off-1).front().meta;
            Meta const &cur = at(i).front().meta;
            if (std::abs(prev[Meta::SLICE_LOCATION] - cur[Meta::SLICE_LOCATION]) <= LOCATION_GAP_EPSILON) {
                LOG(WARNING) << "replacing " << at(off-1).dir()
                             << " (" << prev[Meta::SERIES_NUMBER] << ":" << prev[Meta::SLICE_LOCATION] << ") "
                             << " with " << at(i).dir()
                             << " (" << cur[Meta::SERIES_NUMBER] << ":" << cur[Meta::SLICE_LOCATION] << ") ";
                std::swap(at(off-1), at(i));
            }
            else {
                if (off != i) { // otherwise no need to swap
                    std::swap(at(off), at(i));
                }
                ++off;
            }
        }
        if (off != size()) {
            LOG(WARNING) << "study " << study_path << " reduced from " << size() << " to " << off << " series.";
            resize(off);
        }
        return ok;
    }

    void Study::check_regroup () {
        vector<Series> v;
        v.swap(*this);
        for (Series &s: v) {
            unsigned max_nn = 0;
            unordered_map<float, vector<unsigned>> group;
            for (unsigned i = 0; i < s.size(); ++i) {
                auto const &ss = s[i];
                unsigned nn = ss.meta[Meta::NUMBER_OF_IMAGES];
                if (nn > max_nn) {
                    max_nn = nn;
                }
                group[ss.meta[Meta::SLICE_LOCATION]].push_back(i);
            }
            if ((s.size() <= max_nn) && (group.size() <= 1)) {
                emplace_back(std::move(s));
            }
            else { // regroup
                LOG(WARNING) << "regrouping series " << s.dir() << " into " << group.size() << " groups.";
                unsigned i;
                for (auto const &p: group) {
                    emplace_back();
                    back().series_path = s.series_path;
                    back().series_path += fs::path(':' + lexical_cast<string>(i));;
                    for (unsigned j: p.second) {
                        back().push_back(std::move(s[j]));
                    }
                    sort(back().begin(), back().end());
                    ++i;
                }
            }
        }
    }
}

