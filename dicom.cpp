#include <boost/lexical_cast.hpp>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include "adsb2.h"

namespace adsb2 {
    cv::Mat load_dicom (fs::path const &path, Meta *meta) {
        CHECK(meta);
        DcmFileFormat fileformat;
        OFCondition status = fileformat.loadFile(path.c_str());
        CHECK(status.good()) << "error loading dcm file: " << path;
        OFString pixelSpacing;
        status = fileformat.getDataset()->findAndGetOFString(DCM_PixelSpacing, pixelSpacing);
        CHECK(status.good()) << "cannot find pixel spacing: " << path;
        meta->spacing = boost::lexical_cast<float>(pixelSpacing);
        meta->raw_spacing = meta->spacing;

#if 0   // IMPORTANT: regular images do not have DiCOM meta data
        cv::Mat raw = cv::imread(path.native(), -1);
        if (!raw.data) {
            // failed, resort to convert
            ostringstream ss;
            fs::path tmp(fs::unique_path("%%%%-%%%%-%%%%-%%%%.pgm"));
            ss << "convert " << path << " " << tmp.native();
            ::system(ss.str().c_str());
            raw = cv::imread(tmp.native(), -1);
            fs::remove(tmp);
        }
#else
        DicomImage *dcm = new DicomImage(path.c_str());
        CHECK(dcm) << "fail to new DicomImage";
        CHECK(dcm->getStatus() == EIS_Normal) << "fail to load dcm image";
        CHECK(dcm->isMonochrome()) << " only monochrome data supported.";
        CHECK(dcm->getDepth() == 16) << " only 16-bit data supported.";
        CHECK(dcm->getFrameCount() == 1) << " only single-framed dcm supported.";
        cv::Mat raw(dcm->getHeight(), dcm->getWidth(), CV_16U);
        dcm->getOutputData(raw.ptr<uint16_t>(0), raw.total() * sizeof(uint16_t), 16);
        delete dcm;
#endif
        return raw;
    }

}
