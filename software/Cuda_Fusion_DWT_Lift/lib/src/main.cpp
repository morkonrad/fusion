#include "fusion.h"

#include <iostream>
#include <chrono>
#include <cstdint>
#include <assert.h>


#ifdef _VISUAL_DBG_OPENCV_

#include "opencv2/highgui.hpp"

template<typename T>
void build_show_image(const std::vector<T>& img,int rows,int cols,int wait,const std::string& caption)
{
    cv::Mat cv_img_out(rows,cols,CV_8U);
    for(int y=0;y<cv_img_out.rows;y++)
    {
        for(int x=0;x<cv_img_out.cols;x++)
        {
            auto in_pix = img[y*cols+x];

            cv_img_out.at<uchar>(y,x) = (uchar)in_pix;
        }
    }
    cv::namedWindow(caption,CV_WINDOW_NORMAL/*CV_WINDOW_AUTOSIZE*/);
    cv::imshow(caption,cv_img_out);
    cv::waitKey(wait);
}

int main()
{

    // Get visual image
    //const auto src_img_vis = "/home/morkon/Bilder/img_vis_ir/pD/VIS.jpg";
    const auto src_img_vis = "/home/nvidia/Pictures/VIS-2k.jpg";
    const cv::Mat Mat_in_img_v = cv::imread(src_img_vis,CV_LOAD_IMAGE_GRAYSCALE);

    if(Mat_in_img_v.empty()){
        std::cerr<<" Couldn't open image file: "<<src_img_vis<<" ...fixme->EXIT !!!"<<std::endl;
        std::exit(-1);
    }

    // Get IR image
    //const auto src_img_ir = "/home/morkon/Bilder/img_vis_ir/pD/IR.jpg";
    const auto src_img_ir = "/home/nvidia/Pictures/IR-2k.jpg";
    const cv::Mat Mat_in_img_ir = cv::imread(src_img_ir,CV_LOAD_IMAGE_GRAYSCALE);
    if(Mat_in_img_ir.empty()){
        std::cerr<<" Couldn't open image file: "<<src_img_ir<<" ...fixme->EXIT !!!"<<std::endl;
        std::exit(-1);
    }

    if(Mat_in_img_ir.rows!=Mat_in_img_v.rows)return -1;
    if(Mat_in_img_ir.cols!=Mat_in_img_v.cols)return -1;

    cv::Mat cv_img_float_vis;
    Mat_in_img_v.convertTo(cv_img_float_vis,CV_16S);

    cv::Mat cv_img_float_ir;
    Mat_in_img_ir.convertTo(cv_img_float_ir,CV_16S);

    const std::uint16_t rows = Mat_in_img_v.rows;
    const std::uint16_t cols = Mat_in_img_ir.cols;
    const std::uint16_t lvls = 4;
    const std::uint64_t img_size_bytes = rows*cols*sizeof(DATATYPE);

    const std::uint8_t* imgVis = cv_img_float_vis.datastart;
    const std::uint8_t* imgIR = cv_img_float_ir.datastart;


    std::vector<DATATYPE> results_fused;

    FUSE_CUDA::Fusion_DWT fuse(imgVis,imgIR,
                rows,cols,lvls,img_size_bytes);

    const std::uint16_t lvl_id = lvls;
    const bool async = true;
    const int cnt_iter = 100;
    float time_ms = 0;

    for(int i=0;i<cnt_iter;i++)
    {
        auto start = std::chrono::system_clock::now();
        {
            if(async)
            {
                auto ok = fuse.calc_analysis_async(imgVis,imgIR,cols,rows,img_size_bytes);                
                assert(ok==0);
                //visualize
                //auto ptr_v = fuse.get_analysis_vis();
                //auto ptr_ir = fuse.get_analysis_ir();
                //build_show_image(*ptr_v,rows,cols,0,"VIS");
                //build_show_image(*ptr_ir,rows,cols,0,"IR");

            }
            else
            {
                auto ok = fuse.calc_analysis_sync(lvl_id,imgVis,imgIR,cols,rows,img_size_bytes);
                assert(ok==0);
                //visualize
                /*
                std::vector<DATATYPE> analysis_results_vis;
                std::vector<DATATYPE> analysis_results_ir;
                std::vector<DATATYPE> analysis_results_fused;
                ok = fuse.get_analysis_results(analysis_results_vis,analysis_results_ir);
                ok = fuse.get_analysis_fused(analysis_results_fused);
                build_show_image(analysis_results_ir,rows,cols,0,"IR");
                build_show_image(analysis_results_vis,rows,cols,0,"VIS");
                build_show_image(analysis_results_fused,rows,cols,0,"FUSED_COEF");*/
            }

            auto ok = fuse.get_fused_input(results_fused);
            assert(ok==0);
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end-start;
        const auto titer = diff.count()*1e3;
        time_ms += titer;
        std::cout << " Iteration <"<< i+1 << "> exe_time: "<<titer<<" ms"<<std::endl;
        build_show_image(results_fused,rows,cols,500,"FUSED");
    }

    std::cout << "For "<<cnt_iter<<" iterations avg_time is: "<< (float)time_ms/(float)cnt_iter << " ms"<<std::endl;

    return 0;
}
#else

int main(int argc,char** argv)
{
    if(argc<5)return -1;

    const int rows = std::atoi(argv[1]);
    const int cols = std::atoi(argv[2]);
    const int iter = std::atoi(argv[3]);
    const int lvls = std::atoi(argv[4]);

    if(iter<1)return -1;
    if(rows*cols==0)return -1;
    if(lvls<1)return -1;

    //std::cout<<rows<<"-"<<cols<<"-"<<iter<<"-"<<lvls<<std::endl;

    std::vector<DATATYPE> img_vis(rows*cols,11);
    std::vector<DATATYPE> img_ir(rows*cols,12);

    const std::uint64_t img_size_bytes = rows*cols*sizeof(DATATYPE);
    const std::uint8_t* imgVis = (const std::uint8_t*)&img_vis[0];
    const std::uint8_t* imgIR = (const std::uint8_t*)&img_ir[0];

    std::vector<DATATYPE> results_fused;

    FUSE_CUDA::Fusion_DWT fuse( imgVis,imgIR,
                        rows,cols,
                        lvls,
                        img_size_bytes);

    const std::uint16_t lvl_id = lvls;
    const bool async = 0;
    float time_ms = 0;

    for(int i=0;i<iter;i++)
    {
        auto start = std::chrono::system_clock::now();
        {
            if(async)
            {
                auto ok = fuse.calc_analysis_async(imgVis,imgIR,cols,rows,img_size_bytes);
                assert(ok==0);
            }
            else
            {
                auto ok = fuse.calc_analysis_sync(lvl_id,imgVis,imgIR,cols,rows,img_size_bytes);
                assert(ok==0);
            }

            auto ok = fuse.get_fused_input(results_fused);
            assert(ok==0);
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end-start;
        const auto titer = diff.count()*1e3;
        time_ms += titer;
        //std::cout << " Iteration <"<< i+1 << "> exe_time: "<<titer<<" ms"<<std::endl;
    }

    std::cout << "For "<<iter<<" iterations avg_time is: "<< (float)time_ms/(float)iter << " ms"<<std::endl;
    return 0;

}
#endif
