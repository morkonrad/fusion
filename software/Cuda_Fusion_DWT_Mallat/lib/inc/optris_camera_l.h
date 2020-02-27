#ifndef OPTRIS_CAMERA_H
#define OPTRIS_CAMERA_H


#include <cstdint>
#include <memory>
#include <vector>
#include <string>

#include <libirimager/IRImager.h>
#include <libirimager/IRDeviceUVC.h>
#include <libirimager/IRDeviceParams.h>
#include <libirimager/IRImagerClient.h>
#include <libirimager/FramerateCounter.h>



/*!
  \brief Class for data acquisition from Optris bolometers via libirimager
*/
class optris_camera : public evo::IRImagerClient 
{

    public:
        enum class OptrisCameraStatus { CamOK, CamFailed, CamFlagNotOpen};
        OptrisCameraStatus grab_init();
        OptrisCameraStatus grab_frame(void **data);
        OptrisCameraStatus grab_exit();
        optris_camera(const std::string& config_file);
        ~optris_camera();

    protected:
        void onRawFrame(unsigned char* data, int size) override;
        void onThermalFrame(unsigned short* data, unsigned int w, unsigned int h, evo::IRFrameMetadata meta, void* arg) override;
        void onFlagStateChange(evo::EnumFlagState flagstate, void* arg) override;
        void onProcessExit(void *arg) override;

    private:

        //! Pointer to V4L camera device object
         std::unique_ptr<evo::IRDeviceUVC>  m_camera;
        //! Object for Optris camera specific features
        evo::IRImager       m_imager;

        std::vector<std::uint16_t> rawdata;
        std::vector<std::uint16_t> calibrated_data;

        enum class OptrisFlagState{ FlagOpen, FlagClosed, FlagOpening, FlagClosing, FlagError, FlagUnknown};
        OptrisFlagState flag_state;
        int image_drop_counter;
        double _elapsed;
        int    _cntElapsed;
        int _cntFrames;
        evo::FramerateCounter fpsStream;
        
};


#endif
