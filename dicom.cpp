#include <boost/lexical_cast.hpp>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include "adsb2.h"

namespace adsb2 {
    cv::Mat ImageLoader::load_raw (string const &path, Meta *pmeta) {
        Meta meta;
        DcmFileFormat fileformat;
        OFCondition status = fileformat.loadFile(path.c_str());
        CHECK(status.good()) << "error loading dcm file: " << path;
        OFString pixelSpacing;
        status = fileformat.getDataset()->findAndGetOFString(DCM_PixelSpacing, pixelSpacing);
        CHECK(status.good()) << "cannot find pixel spacing: " << path;
        meta.spacing = boost::lexical_cast<float>(pixelSpacing);
        meta.raw_spacing = meta.spacing;

        cv::Mat v;
#ifdef DO_NOT_USE_DICOM
        v = cv::imread(path, -1);
        if (!v.data) {
            // failed, resort to convert
            ostringstream ss;
            fs::path tmp(fs::unique_path("%%%%-%%%%-%%%%-%%%%.pgm"));
            ss << "convert " << path << " " << tmp.native();
            ::system(ss.str().c_str());
            v = cv::imread(tmp.native(), -1);
            fs::remove(tmp);
        }
        if (!v.data) return v;
#else
        DicomImage *dcm = new DicomImage(path.c_str());
        if (!dcm) return v;
        CHECK(dcm->getStatus() == EIS_Normal) << "fail to load dcm image";
        CHECK(dcm->isMonochrome()) << " only monochrome data supported.";
        CHECK(dcm->getDepth() == 16) << " only 16-bit data supported.";
        CHECK(dcm->getFrameCount() == 1) << " only single-framed dcm supported.";
        v.create(dcm->getHeight(), dcm->getWidth(), CV_16U);
        dcm->getOutputData(v.ptr<uint16_t>(0), v.total() * sizeof(uint16_t), 16);
        delete dcm;
#endif
        if (pmeta) {
            *pmeta = meta;
        }
        return v;
    }

}
