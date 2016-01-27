#include <boost/lexical_cast.hpp>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include "adsb2.h"

namespace adsb2 {

    void dicom_setup (char const *path, Config const &config) {
        fs::path def = fs::path(path).parent_path() / fs::path("dicom.dic");
        string v = config.get<string>("adsb2.dcmdict", def.native());
        setenv("DCMDICTPATH", v.c_str(), 0);
    }

    template <typename T>
    T dicom_get (DcmFileFormat &ff, DcmTagKey key, fs::path const &path) {
        OFString str;
        OFCondition status = ff.getDataset()->findAndGetOFString(key, str);
        CHECK(status.good()) << "cannot find element " << key << ": " << path;
        return boost::lexical_cast<T>(str);
    }

    template <>
    string dicom_get (DcmFileFormat &ff, DcmTagKey key, fs::path const &path) {
        OFString str;
        OFCondition status = ff.getDataset()->findAndGetOFString(key, str);
        CHECK(status.good()) << "cannot find element " << key << ": " << path;
        return string(str.begin(), str.end());
    }
    /*
        struct Series {
            float slice_thickness;  // mm
            float slice_spacing;    // mm
            float nominal_interval;
            float repetition_time;
            float echo_time;
            int number_of_images;
        };
        struct Slice {
            float trigger_time;
            int series_number;
            float slice_location;
        };
        */

    cv::Mat load_dicom (fs::path const &path, Meta *meta) {
        CHECK(meta);
        DcmFileFormat ff;
        OFCondition status = ff.loadFile(path.c_str());
        CHECK(status.good()) << "error loading dcm file: " << path;
        meta->study.part = dicom_get<string>(ff, DCM_BodyPartExamined, path);
        CHECK(meta->study.part == "HEART");
        string sex = dicom_get<string>(ff, DCM_PatientSex, path);
        CHECK(sex == "F" || sex == "M");
        meta->study.sex = sex[0];
        string age = dicom_get<string>(ff, DCM_PatientAge, path);
        CHECK(age.back() == 'Y');
        age.pop_back();
        meta->study.age = boost::lexical_cast<float>(age);
        meta->series.slice_thickness = dicom_get<float>(ff, DCM_SliceThickness, path);
        //meta->series.slice_spacing = dicom_get<float>(ff, DCM_SpacingBetweenSlices, path);
        meta->series.nominal_interval = dicom_get<float>(ff, DCM_NominalInterval, path);
        meta->series.repetition_time = dicom_get<float>(ff, DCM_RepetitionTime, path);
        meta->series.echo_time = dicom_get<float>(ff, DCM_EchoTime, path);
        meta->series.slice_location = dicom_get<float>(ff, DCM_SliceLocation, path);
        meta->series.number_of_images = dicom_get<int>(ff, DCM_CardiacNumberOfImages, path);
        meta->series.series_number = dicom_get<int>(ff, DCM_SeriesNumber, path);
        meta->trigger_time = dicom_get<float>(ff, DCM_TriggerTime, path);
        meta->spacing = dicom_get<float>(ff, DCM_PixelSpacing, path);
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
