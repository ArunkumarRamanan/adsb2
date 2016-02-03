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
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>
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

    fs::path home_dir;
    void GlobalInit (char const *path, Config const &config) {
        FLAGS_logtostderr = 1;
        FLAGS_minloglevel = 1;
        home_dir = fs::path(path).parent_path();
        google::InitGoogleLogging(path);
        dicom_setup(path, config);
        //openblas_set_num_threads(config.get<int>("adsb2.threads.openblas", 1));
        cv::setNumThreads(config.get<int>("adsb2.threads.opencv", 1));
    }

    BoxAnnoOps box_anno_ops;
    PolyAnnoOps poly_anno_ops;
    
    void BoxAnnoOps::load (Slice *slice, string const *txt) const
    {
        slice->anno = this;
        Data &box = slice->anno_data.box;
        box.x = lexical_cast<float>(txt[0]);
        box.y = lexical_cast<float>(txt[1]);
        box.width = lexical_cast<float>(txt[2]);
        box.height = lexical_cast<float>(txt[3]);
    }

    void BoxAnnoOps::shift (Slice *slice, cv::Point_<float> const &pt) const {
        Data &box = slice->anno_data.box;
        box.x -= pt.x;
        box.y -= pt.y;
    }

    void BoxAnnoOps::scale (Slice *slice, float rate) const {
        Data &box = slice->anno_data.box;
        box.x *= rate;
        box.y *= rate;
        box.width *= rate;
        box.height *= rate;
    }

    void BoxAnnoOps::fill (Slice const &slice, cv::Mat *out, cv::Scalar const &v) const
    {
        Data const &box = slice.anno_data.box;
#define BOX_AS_CIRCLE 1
#ifdef BOX_AS_CIRCLE
        cv::circle(*out, round(cv::Point_<float>(box.x + box.width/2, box.y + box.height/2)),
                         std::sqrt(box.width * box.height)/2, v, CV_FILLED);
        // TODO, use rotated rect to draw ellipse 
#else
        cv::rectangle(*out, round(box), v, CV_FILLED);
        
#endif
    }

    void PolyAnnoOps::load (Slice *slice, string const *txt) const
    {
        CHECK(0);
    }

    void PolyAnnoOps::shift (Slice *slice, cv::Point_<float> const &pt) const {
        CHECK(0);
    }

    void PolyAnnoOps::scale (Slice *slice, float rate) const {
        CHECK(0);
    }

    void PolyAnnoOps::fill (Slice const &, cv::Mat *, cv::Scalar const &) const
    {
        CHECK(0);
    }


    Slice::Slice (string const &txt)
        : do_not_cook(false),
        pred_box(-1,-1,0,0),
        pred_box_reliable(false),
        pred_area(-1)
    {
        using namespace boost::algorithm;
        line = txt;
        vector<string> ss;
        split(ss, line, is_any_of("\t"), token_compress_off);
        path = fs::path(ss[0]);

        string const *rest = &ss[1];
        int nf = ss.size() - 1;

        if (nf == 3) {
            LOG(ERROR) << "annotation format not supported.";
            CHECK(0);
        }
        else if (nf == 4) {
            box_anno_ops.load(this, rest);
        }
        else if (nf == 5) {
            LOG(ERROR) << "annotation format not supported.";
            CHECK(0);
        }
        else if (nf >= 7) {
            poly_anno_ops.load(this, rest);
        }
        else {
            LOG(ERROR) << "annotation format not supported.";
            CHECK(0);
        }
    }

    void Slice::clone (Slice *s) const {
        s->id = id;
        s->path = path;
        s->meta = meta;
        s->raw = raw.clone();
        s->image = image.clone();
        s->vimage = vimage.clone();
        s->do_not_cook = do_not_cook;
        s->line = line;
        s->anno = anno;
        s->anno_data = anno_data;
        s->prob = prob.clone();
        s->pred_box = pred_box;
        s->pred_area = pred_area;
    }

    void Slice::visualize (bool show_prob) {
        CHECK(image.type() == CV_32FC1);
#if 0
        cv::Mat rgb;
        cv::cvtColor(image, rgb, CV_GRAY2BGR, 0);
        rgb.convertTo(image, CV_8UC3);
        cv::Scalar color = pred_box_reliable ? cv::Scalar(0, 0xFF, 0) : cv::Scalar(0, 0, 0xFF);
#else
        cv::Scalar color(0xFF);
#endif
        if (pred_box.x >= 0) {
            cv::rectangle(image, pred_box, color);
        }
        if (show_prob && prob.data) {
            cv::Mat pp;
            cv::normalize(prob, pp, 0, 255, cv::NORM_MINMAX, CV_8U);
#if 0
            cv::cvtColor(pp, rgb, CV_GRAY2BGR);
            if (pred_box.x >= 0) {
                cv::rectangle(rgb, pred_box, color);
            }
            cv::hconcat(image, rgb, image);
#else
            if (pred_box.x >= 0) {
                cv::rectangle(pp, pred_box, color);
            }
            cv::hconcat(image, pp, image);
#endif
        }
    }

    void Slice::update_polar (cv::Point_<float> const &C, float R, Detector *det) {
        polar_C = C;
        polar_R = R;

        cv::Mat polar;
        linearPolar(image, &polar, polar_C, polar_R, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
        if (!det) {
            polar_prob = polar;
        }
        else {
            int m = polar.rows / 4;
            cv::Mat extended;
            vconcat3(polar.rowRange(polar.rows - m, polar.rows),
                     polar,
                     polar.rowRange(0, m),
                     &extended);
            cv::Mat extended_prob;
            det->apply(extended, &extended_prob);
            polar_prob = extended_prob.rowRange(m, m + polar.rows).clone();
            polar_prob *= 255;
        }
    }

#if 0
    void Slice::eval (cv::Mat mat, float *s1, float *s2) const {
        CHECK(box.x >=0 && box.y >= 0);
        cv::Mat roi = mat(round(box));
        float total = cv::sum(mat)[0];
        float covered = cv::sum(roi)[0];
        *s1 = covered / total;
        *s2 = 0;//*s2 = std::sqrt(roi.area()) * meta.spacing;
    }
#endif

    Series::Series (fs::path const &path_, bool load, bool check, bool fix): path(path_) {
        // enumerate DCM files
        vector<fs::path> paths;
        fs::directory_iterator end_itr;
        for (fs::directory_iterator itr(path);
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

    void Series::shrink (cv::Rect const &bb) {
        CHECK(size());
        CHECK(bb.x >= 0);
        CHECK(bb.y >= 0);
        CHECK(bb.width < front().image.cols);
        CHECK(bb.height < front().image.rows);
        cv::Mat vimage = front().vimage(bb).clone();
        for (Slice &s: *this) {
            s.image = s.image(bb).clone();
            s.vimage = vimage;
            if (s.anno) {
                s.anno->shift(&s, unround(bb.tl()));
            }
            CHECK(s.pred_box.x < 0);
            CHECK(s.pred_box.y < 0);
            CHECK(!s.prob.data);
        }
    }

    void Series::save_dir (fs::path const &dir, fs::path const &ext) {
        fs::create_directories(dir);
        for (auto const &s: *this) {
            CHECK(s.image.depth() == CV_8U) << "image not suitable for visualization, call visualize() first";
            fs::path path(dir / s.path.stem());
            path += ext;
            cv::imwrite(path.native(), s.image);
        }
    }

    void Series::save_gif (fs::path const &path) {
        fs::path tmp(fs::unique_path());
        fs::create_directories(tmp);
        ostringstream gif_cmd;
        gif_cmd << "convert -delay 5 ";
        fs::path pgm(".pgm");
        fs::path pbm(".pbm");
        for (auto const &s: *this) {
            CHECK(s.image.depth() == CV_8U) << "image not suitable for visualization, call visualize() first";
            fs::path pnm(tmp / s.path.stem());
            if (s.image.channels() == 1) {
                pnm += pgm;
            }
            else if (s.image.channels() == 3) {
                pnm += pbm;
            }
            else {
                CHECK(0) << "image depth not supported.";
            }
            cv::imwrite(pnm.native(), s.image);
            gif_cmd << " " << pnm;
        }
        gif_cmd << " " << path;
        ::system(gif_cmd.str().c_str());
        fs::remove_all(tmp);
    }

    void Series::visualize (bool show_prob) {
        for (Slice &s: *this) {
            s.visualize(show_prob);
        }
    }

    void Series::getVImage (cv::Mat *vimage) {
        if (size() <= 1) {
            *vimage = cv::Mat(front().image.size(), CV_32F, cv::Scalar(0));
            return;
        }
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
        *vimage = sigma;
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

    Study::Study (fs::path const &path_, bool load, bool check, bool fix): path(path_) {
        // enumerate DCM files
        vector<fs::path> paths;
        fs::directory_iterator end_itr;
        for (fs::directory_iterator itr(path);
                itr != end_itr; ++itr) {
            if (fs::is_directory(itr->status())) {
                // found subdirectory,
                // create tagger
                auto sax = itr->path();
                string name = sax.filename().native();
                if (name.find("sax_") != 0) {
                    continue;
                }
                paths.push_back(sax);
            }
        }
        std::sort(paths.begin(), paths.end());
        CHECK(paths.size());
        for (auto const &sax: paths) {
            emplace_back(sax, load, false, false);    // do not fix for now
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
                LOG(WARNING) << "Study " << path << " series " << s.path << " sanity check failed.";
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
            LOG(WARNING) << "study " << path << " reduced from " << size() << " to " << off << " series.";
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
                    back().path = s.path;
                    back().path += fs::path(':' + lexical_cast<string>(i));;
                    for (unsigned j: p.second) {
                        back().push_back(std::move(s[j]));
                    }
                    sort(back().begin(), back().end());
                    ++i;
                }
            }
        }
    }

    // histogram equilization
    void getColorMap (Series const &series, vector<float> *cmap, int colors) {
        vector<uint16_t> all;
        all.reserve(series.front().image.total() * series.size());
        for (auto const &s: series) {
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

    void Cook::apply (Slice *slice) const {
        CHECK(0) << "Unimplemented";   // not supported yet
    }

    void Cook::apply (Series *series) const {
        // normalize color
        cv::Mat vimage;
        series->getVImage(&vimage);
        cv::Size raw_size = vimage.size();
        // compute var image
        cv::normalize(vimage, vimage, 0, color_bins-1, cv::NORM_MINMAX, CV_32FC1);
        float scale = -1;
        float raw_spacing = -1;
        cv::Size sz;
        if (spacing > 0) {
            //float scale = spacing / meta.spacing;
            raw_spacing = series->front().meta.raw_spacing;
            scale = raw_spacing / spacing;
            sz = round(raw_size * scale);
            cv::resize(vimage, vimage, sz);
        }
        vector<float> cmap;
        getColorMap(*series, &cmap, color_bins);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < series->size(); ++i) {
            auto &s = series->at(i);
            if (s.do_not_cook) continue;
            equalize(s.raw, &s.image, cmap);
            if (scale > 0) {
                s.meta.spacing = spacing;
                CHECK(s.meta.raw_spacing == raw_spacing);
                CHECK(s.image.size() == raw_size);
                //float scale = s.meta.raw_spacing / s.meta.spacing;
                cv::resize(s.image, s.image, sz);
                if (s.anno) {
                    s.anno->scale(&s, scale);
                }
            }
#pragma omp critical
            s.vimage = vimage;
        }
    }

    void Cook::apply (Study *study) const {
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
                to.anno = from.anno;
                to.anno_data = from.anno_data;
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

    float accumulate (cv::Mat const &image, vector<float> *pX, vector<float> *pY) {
        vector<float> X(image.cols, 0);
        vector<float> Y(image.rows, 0);
        float total = 0;
        CHECK(image.type() == CV_32F);
        for (int y = 0; y < image.rows; ++y) {
            float const *row = image.ptr<float const>(y);
            for (int x = 0; x < image.cols; ++x) {
                float v = row[x];
                X[x] += v;
                Y[y] += v;
                total += v;
            }
        }
        pX->swap(X);
        pY->swap(Y);
        return total;
    }

    Eval::Eval () {
        ifstream is("train.csv");
        CHECK(is);
        string dummy;
        getline(is, dummy);
        int a;
        char d1, d2;
        for (unsigned i = 0; i < 500; ++i) {
            is >> a >> d1 >> volumes[i][0] >> d2 >> volumes[i][1];
            CHECK(a == i+1);
            CHECK(d1 == ',');
            CHECK(d2 == ',');
        }
    }

    float Eval::crps (float v, vector<float> const &x) {
        CHECK(x.size() == VALUES);
        float sum = 0;
        unsigned i = 0;
        for (; i < v; ++i) {
            float s = x[i];
            sum += s * s;
        }
        for (; i < VALUES; ++i) {
            float s = 1.0 - x[i];
            sum += s * s;
        }
        for (unsigned i = 0; i < VALUES; ++i) {
            CHECK(x[i] >= 0);
            CHECK(x[i] <= 1);
            if (i > 0) CHECK(x[i] >= x[i-1]);
        }
        return sum/VALUES;
    }

    float Eval::score (fs::path const &path, vector<std::pair<string, float>> *s) {
        fs::ifstream is(path);
        string line;
        getline(is, line);
        float sum = 0;
        s->clear();
        while (getline(is, line)) {
            using namespace boost::algorithm;
            vector<string> ss;
            split(ss, line, is_any_of(",_"), token_compress_on);
            CHECK(ss.size() == VALUES + 2);
            string name = ss[0] + "_" + ss[1];
            int n = lexical_cast<int>(ss[0]) - 1;
            CHECK(n >= 0 && n < CASES);
            int m;
            if (ss[1] == "Systole") {
                m = 0;
            }
            else if (ss[1] == "Diastole") {
                m = 1;
            }
            else CHECK(0);
            float v = volumes[n][m];
            vector<float> x;
            for (unsigned i = 0; i < VALUES; ++i) {
                x.push_back(lexical_cast<float>(ss[2 + i]));
            }
            float score = crps(v, x);
            s->push_back(std::make_pair(name, score));
            sum += score;
        }
        CHECK(s->size());
        return sum / s->size();
    }
    float Eval::score (unsigned n1, unsigned n2, vector<float> const &x) {
        --n1;
        CHECK(n1 >= 0 && n1 < CASES);
        float v = volumes[n1][n2];
        return crps(v, x);
    }
}

