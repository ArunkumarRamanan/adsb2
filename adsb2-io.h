#pragma once

namespace adsb2 {
    namespace io {
        using std::istream;
        using std::ostream;

        template <typename T>
        static inline void write (ostream &os, T const &v) {
            os.write(reinterpret_cast<char const *>(&v), sizeof(T));
        }

        template <typename T>
        static inline void read (istream &is, T *v) {
            is.read(reinterpret_cast<char *>(v), sizeof(T));
        }

        template <typename T>
        static inline void write (ostream &os, vector<T> const &v) {
            size_t sz = v.size();
            write(os, sz);
            if (sz > 0) {
                os.write(reinterpret_cast<char const *>(&v[0]), sizeof(T) * v.size());
            }
        }

        template <typename T>
        static inline void read (istream &is, vector<T> *v) {
            size_t sz;
            read(is, &sz);
            v->resize(sz);
            if (sz > 0) {
                is.read(reinterpret_cast<char *>(&v->at(0)), sizeof(T) * sz);
            }
        }

        static inline void write (ostream &os, std::string const &s) {
            size_t sz = s.size();
            write(os, sz);
            if (sz > 0) {
                os.write(&s[0], sz);
            }
        }

        static inline void read (istream &is, std::string *s) {
            size_t sz;
            read(is, &sz);
            s->resize(sz);
            if (sz > 0) {
                is.read(&(*s)[0], sz);
            }
        }

        static inline void write (ostream &os, boost::filesystem::path const &path) {
            std::string n = path.native();
            write(os, n);
        }

        static inline void read (istream &is, boost::filesystem::path *path) {
            std::string n;
            read(is, &n);
            *path = boost::filesystem::path(n);
        }

        static inline void write (ostream &os, cv::Mat mat) {
            if (!mat.isContinuous()) {
                mat = mat.clone();
            }
            int rows = mat.rows;
            int cols = mat.cols;
            int type = mat.type();
            write(os, rows);
            write(os, cols);
            write(os, type);
            if (mat.total() > 0) {
                os.write(reinterpret_cast<char const *>(mat.data), mat.total() * mat.elemSize());
            }
            /*
            cv::FileStorage sr("dummy.yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

            sr << "M" << mat;
            std::string buf = sr.releaseAndGetString();
            write(os, buf);
            */
        }

        static inline void read (istream &is, cv::Mat *mat) {
            int rows, cols, type;
            read(is, &rows);
            read(is, &cols);
            read(is, &type);
            mat->create(rows, cols, type);
            if (mat->total() > 0) {
                is.read(reinterpret_cast<char *>(mat->data), mat->total() * mat->elemSize());
            }

            /*
            std::string buf;
            read(is, &buf);
            cv::FileStorage sr(buf, cv::FileStorage::READ | cv::FileStorage::MEMORY);
            sr["M"] >> *mat;
            */
        }
    }
}
