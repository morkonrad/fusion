#include "f_Lap_pyr.h"


void f_Lap_pyr::calcfilter(const float &sigma, const ePyramidConvolution &conv)
{

    float M_PI_= 22 / 7;

    double s = 2.0 * sigma * sigma, sum = 0;

    if (conv == ePyramidConvolution::Gauss_separable)
    {

        int frad = (int)(ceil(3.5f * sigma));

        filter_len = 2 * frad + 1;

        h_filter = new float[filter_len];

        for (int x = -frad; x < 0; x++)
        {

            sum += h_filter[x + frad] = h_filter[abs(x - frad)] = float((exp(-(x*x) / s)) / sqrt(M_PI_ * s));

        }
        h_filter[frad] = float(1 / sqrt(M_PI_ * s));
        sum = 2 * sum + h_filter[frad];

        h_filter[frad] /= float(sum);
        // normalize the Kernel
        for (int x = -frad; x < 0; x++) {

            float a = h_filter[x + frad];
            a /= float(sum);
            h_filter[x + frad] = h_filter[abs(x - frad)] = a;

        }
    }

    if (conv == ePyramidConvolution::Gauss_recursive)
    {
        filter_len = 8;

        h_filter = new float[filter_len];

        const float
                nsigma = sigma < 0.1f ? 0.1f : sigma,
                alpha = 1.695f / nsigma,
                ema = (float)std::exp(-alpha),
                ema2 = (float)std::exp(-2 * alpha);

        const float k = (1 - ema)*(1 - ema) / (1 + 2 * alpha*ema - ema2);

        h_filter[0] = k;	//a0
        h_filter[1] = k*(alpha - 1)*ema;//a1
        h_filter[2] = k*(alpha + 1)*ema;//a2
        h_filter[3] = -k*ema2;//a3
        h_filter[4] = -2 * ema,//b1
                h_filter[5] = ema2;//b2

        h_filter[6] = (h_filter[0] + h_filter[1]) / (1 + h_filter[4] + h_filter[5]);//coefp
        h_filter[7] = (h_filter[2] + h_filter[3]) / (1 + h_filter[4] + h_filter[5]);//coefn
    }


}

void f_Lap_pyr::len_calc()
{
    out_len = 0;
    length = new int* [scales + 1];

    for (int i = 0; i < scales + 1; i++)
    {
        length[i] = new int[2];
    }

    int J = scales;

    int r = img_rows, c = img_cols;

    length[J][0] = r;
    length[J][1] = c;

    signal_len = r*c;

    out_len += r*c;
    J--;

    while (J >= 0) {

        r = (int)ceil((double)r / 2.0);
        c = (int)ceil((double)c / 2.0);

        length[J][0] = r;
        length[J][1] = c;

        out_len += r*c;

        J--;
    }

}

void f_Lap_pyr::set_n_lev(int n_levels)
{
    scales = n_levels;

    len_calc();

    checkCudaErrors(cudaMalloc((void **)&d_lap_v, out_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_lap_ir, out_len * sizeof(float)));

}

void f_Lap_pyr::filt_alloc(const float &sigma, const ePyramidConvolution &conv) {

    conv_methd = conv;

    calcfilter(sigma, conv);

    checkCudaErrors(cudaMalloc((void **)&d_filter, filter_len * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_filter, h_filter, filter_len * sizeof(float), cudaMemcpyHostToDevice));

    set_filter(d_filter);

}

void f_Lap_pyr::filt_free() {

    free(h_filter);
    cudaFree(d_filter);
    filter_len = 0;

}

void f_Lap_pyr::LP_init(int rows, int cols, float sigma, int n_levels, ePyramidConvolution conv_method)
{
    img_rows = rows; img_cols = cols;

    filt_alloc(sigma, conv_method);

    set_n_lev(n_levels);

    //Device memories for respective intermediate results
    checkCudaErrors(cudaMalloc((void **)&d_input_ir, signal_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_reduce_ir, signal_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_buffer_ir, signal_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_expand_ir, signal_len * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_input_v, signal_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_reduce_v, signal_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_buffer_v, signal_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_expand_v, signal_len * sizeof(float)));

    //Pinned host memory to store Fused image
    checkCudaErrors(cudaHostAlloc((void**)&fused, signal_len * sizeof(float), cudaHostAllocDefault));

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    ROI = {cols, rows};

    //Color
    Image_V_C = nppiMalloc_8u_C4(cols, rows, &step_C);
    Image_IR_C = nppiMalloc_8u_C4(cols, rows, &step_C);

    //Gray
    Image_V_Gr = nppiMalloc_8u_C1(cols, rows, &step_G);
    Image_IR_Gr = nppiMalloc_8u_C1(cols, rows, &step_G);

    //Floating point
    Image_V_F = nppiMalloc_32f_C1(cols, rows, &step_F);
    Image_IR_F = nppiMalloc_32f_C1(cols, rows, &step_F);

}

void CU_Sep_conv_down_smp_1(float *d_ip, float *d_Buffer, float *d_red, const int &filt_len, int **length, const int &iter, const cudaStream_t &stream) {

    int filter_Rad = filt_len / 2;

    int rows = length[iter][0], cols = length[iter][1];

    int cA_rows = length[iter - 1][0], cA_cols = length[iter - 1][1];

    int Halo_steps = filter_Rad / ROWS_BLOCKDIM_X + 1;
    if (filter_Rad % ROWS_BLOCKDIM_X == 0)	Halo_steps--;

    convolutionRowsGPU_down_smp(d_Buffer, d_ip, cols, cA_cols, rows, filter_Rad, Halo_steps, stream);

    Halo_steps = filter_Rad / COLUMNS_BLOCKDIM_Y + 1;
    if (filter_Rad % COLUMNS_BLOCKDIM_Y == 0)    Halo_steps--;

    convolutionColumnsGPU_down_smp(d_red, d_Buffer, cA_cols, rows, cA_rows, filter_Rad, Halo_steps, stream);
}

void CU_Rec_conv_down_smp_1(float *d_ip, float *d_Buffer, float *d_red, const int &filt_len, int **length, const int &iter,const cudaStream_t &stream) {

    int filter_Rad = filt_len / 2;

    int rows = length[iter][0], cols = length[iter][1];

    int cA_rows = length[iter - 1][0], cA_cols = length[iter - 1][1];

    convolutionRecursive_down_smp_1(d_red, d_ip, d_Buffer, cols, rows, cA_rows, cA_cols,stream);

}

void CU_Sep_conv_up_smp_1(float *d_ip, float *d_Buffer, float *d_exp, const int &filt_len, int **length, const int &iter,cudaStream_t stream) {

    int filter_Rad = filt_len / 2;

    int rows = length[iter - 1][0], cols = length[iter - 1][1];

    int next_rows = length[iter][0], next_cols = length[iter][1];

    int Halo_steps = filter_Rad / ROWS_BLOCKDIM_X + 1;
    if (filter_Rad % ROWS_BLOCKDIM_X == 0)	Halo_steps--;

    convolutionRowsGPU_up_smp(d_Buffer, d_ip, cols, next_cols, rows, filter_Rad, Halo_steps, stream);

    Halo_steps = filter_Rad / COLUMNS_BLOCKDIM_Y + 1;
    if (filter_Rad % COLUMNS_BLOCKDIM_Y == 0)    Halo_steps--;

    convolutionColumnsGPU_up_smp(d_exp, d_Buffer, next_cols, rows, next_rows, filter_Rad, Halo_steps, stream);

}

void CU_Rec_conv_up_smp_1(float *d_ip, float *d_Buffer, float *d_exp, const int &filt_len, int **length, const int &iter,cudaStream_t stream) {

    int filter_Rad = filt_len / 2;

    int rows = length[iter - 1][0], cols = length[iter - 1][1];

    int next_rows = length[iter][0], next_cols = length[iter][1];

    convolutionRecursive_up_smp_1(d_exp, d_ip, d_Buffer, cols, rows, next_cols, next_rows,stream);

}

void f_Lap_pyr::Laplacian_const_Sep() {


    int len = 0, pos_l = out_len;
    int avg_len = length[0][0] * length[0][1];

    int i = scales;
    while (i > 0)
    {
        //convolve and downsample
        CU_Sep_conv_down_smp_1(d_input_v, d_buffer_v, d_reduce_v, filter_len, length, i, stream1);
        CU_Sep_conv_down_smp_1(d_input_ir, d_buffer_ir, d_reduce_ir, filter_len, length, i, stream2);


//                cv::Mat dbg_ir(img_rows/2,img_cols/2,CV_32FC1);
//                cv::Mat dbg_vis(img_rows/2,img_cols/2,CV_32FC1);

//                auto sizebytes = img_rows/2*img_cols/2  * sizeof(float);
//                auto err = cudaMemcpy(dbg_vis.data, d_reduce_v, sizebytes,cudaMemcpyDeviceToHost);
//                assert(err==0);

//                err = cudaMemcpy(dbg_ir.data, d_reduce_ir, sizebytes,cudaMemcpyDeviceToHost);
//                assert(err==0);

//                dbg_vis.convertTo(dbg_vis, CV_8U);
//                cv::imshow("vis_DBG",dbg_vis);
//                cv::waitKey();

//                dbg_ir.convertTo(dbg_ir, CV_8U);
//                cv::imshow("ir_DBG",dbg_ir);
//                cv::waitKey();


        //convolve and upsample
        CU_Sep_conv_up_smp_1(d_reduce_v, d_buffer_v, d_expand_v, filter_len, length, i, stream1);
        CU_Sep_conv_up_smp_1(d_reduce_ir, d_buffer_ir, d_expand_ir, filter_len, length, i, stream2);

        len = length[i][0] * length[i][1];

        pos_l -= len;

        //build differences
        CU_subtract(d_lap_v + pos_l, d_input_v, d_expand_v, len, stream1);
        CU_subtract(d_lap_ir + pos_l, d_input_ir, d_expand_ir, len, stream2);

        i--;

        if (i >= 1)
        {

            CU_Sep_conv_down_smp_1(d_reduce_v, d_buffer_v, d_input_v, filter_len, length, i, stream1);
            CU_Sep_conv_down_smp_1(d_reduce_ir, d_buffer_ir, d_input_ir, filter_len, length, i, stream2);

            CU_Sep_conv_up_smp_1(d_input_v, d_buffer_v, d_expand_v, filter_len, length, i, stream1);
            CU_Sep_conv_up_smp_1(d_input_ir, d_buffer_ir, d_expand_ir, filter_len, length, i, stream2);

            len = length[i][0] * length[i][1];

            pos_l -= len;

            CU_subtract(d_lap_v + pos_l, d_reduce_v, d_expand_v, len, stream1);
            CU_subtract(d_lap_ir + pos_l, d_reduce_ir, d_expand_ir, len, stream2);

            i--;
        }

        //Averaging last level coefficients
        if (i == 0)
        {

            if (scales % 2 != 0) {

                checkCudaErrors(cudaMemcpy(d_lap_v, d_reduce_v, avg_len * sizeof(float), cudaMemcpyDeviceToDevice));
                checkCudaErrors(cudaMemcpy(d_lap_ir, d_reduce_ir, avg_len * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            else {

                checkCudaErrors(cudaMemcpy(d_lap_v, d_input_v, avg_len * sizeof(float), cudaMemcpyDeviceToDevice));
                checkCudaErrors(cudaMemcpy(d_lap_ir, d_input_ir, avg_len * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }

    }

    compare_Lp(d_lap_v, d_lap_ir, avg_len, out_len, stream1,stream2);

}

void f_Lap_pyr::Image_reconst_Sep() {

    int len = 0;

    for (int i = 1; i <= scales; i++)
    {
        if (i == 1) CU_Sep_conv_up_smp_1(d_lap_v, d_buffer_v, d_expand_v, filter_len, length, i, stream1);

        else CU_Sep_conv_up_smp_1(d_expand_v, d_buffer_v, d_expand_v, filter_len, length, i, stream2);

        int next_len = length[i][0] * length[i][1];

        len += length[i - 1][0] * length[i - 1][1];

        CU_add(d_expand_v, d_lap_v + len, next_len);
    }

    checkCudaErrors(cudaMemcpyAsync(fused, d_expand_v, signal_len * sizeof(float), cudaMemcpyDeviceToHost,stream1));
    cudaDeviceSynchronize();
}

void f_Lap_pyr::Laplacian_const_Rec() {


    int len = 0, pos_l = out_len;
    int avg_len = length[0][0] * length[0][1];

    int i = scales;
    while (i > 0)
    {
        CU_Rec_conv_down_smp_1(d_input_v, d_buffer_v, d_reduce_v, filter_len, length, i, stream1);
        CU_Rec_conv_down_smp_1(d_input_ir, d_buffer_ir, d_reduce_ir, filter_len, length, i, stream2);

        CU_Rec_conv_up_smp_1(d_reduce_v, d_buffer_v, d_expand_v, filter_len, length, i, stream1);
        CU_Rec_conv_up_smp_1(d_reduce_ir, d_buffer_ir, d_expand_ir, filter_len, length, i, stream2);

        len = length[i][0] * length[i][1];

        pos_l -= len;

        CU_subtract(d_lap_v + pos_l, d_input_v, d_expand_v, len, stream1);
        CU_subtract(d_lap_ir + pos_l, d_input_ir, d_expand_ir, len, stream2);

        i--;

        if(i>=1)
        {
            CU_Rec_conv_down_smp_1(d_reduce_v, d_buffer_v, d_input_v, filter_len, length, i, stream1);
            CU_Rec_conv_down_smp_1(d_reduce_ir, d_buffer_ir, d_input_ir, filter_len, length, i, stream2);

            CU_Rec_conv_up_smp_1(d_input_v, d_buffer_v, d_expand_v, filter_len, length, i, stream1);
            CU_Rec_conv_up_smp_1(d_input_ir, d_buffer_ir, d_expand_ir, filter_len, length, i, stream2);

            len = length[i][0] * length[i][1];

            pos_l -= len;

            CU_subtract(d_lap_v + pos_l, d_reduce_v, d_expand_v, len, stream1);
            CU_subtract(d_lap_ir + pos_l, d_reduce_ir, d_expand_ir, len, stream2);

            i--;
        }

        //Averaging last level coefficients
        if (i == 0) {

            if (scales % 2 != 0) {

                checkCudaErrors(cudaMemcpyAsync(d_lap_v, d_reduce_v, avg_len * sizeof(float), cudaMemcpyDeviceToDevice, stream1));
                checkCudaErrors(cudaMemcpyAsync(d_lap_ir, d_reduce_ir, avg_len * sizeof(float), cudaMemcpyDeviceToDevice, stream2));
            }
            else {

                checkCudaErrors(cudaMemcpyAsync(d_lap_v, d_input_v, avg_len * sizeof(float), cudaMemcpyDeviceToDevice, stream1));
                checkCudaErrors(cudaMemcpyAsync(d_lap_ir, d_input_ir, avg_len * sizeof(float), cudaMemcpyDeviceToDevice, stream2));
            }
        }

    }

    compare_Lp(d_lap_v, d_lap_ir, avg_len, out_len, stream1, stream2);

}

void f_Lap_pyr::Image_reconst_Rec() {

    int len = 0;

    for (int i = 1; i <= scales; i++)
    {
        if (i == 1)
            CU_Rec_conv_up_smp_1(d_lap_v, d_buffer_v, d_expand_v, filter_len, length, i, stream1);
        else
            CU_Rec_conv_up_smp_1(d_expand_v, d_buffer_v, d_expand_v, filter_len, length, i, stream1);

        int next_len = length[i][0] * length[i][1];

        len += length[i - 1][0] * length[i - 1][1];

        CU_add(d_expand_v, d_lap_v + len, next_len);
    }

    checkCudaErrors(cudaMemcpyAsync(fused, d_expand_v, signal_len * sizeof(float), cudaMemcpyDeviceToHost, stream1));
    cudaDeviceSynchronize();
}

//float* f_Lap_pyr::Fuse_Lap_Pyr_Grayscale(const QImage& IP_Img_1, const QImage& IP_Img_2)
//{

//    uchar *img_v, *img_ir;
//    img_v = (uchar*)IP_Img_1.bits();
//    img_ir = (uchar*)IP_Img_2.bits();

//    cudaHostRegister(img_v, signal_len * sizeof(uchar), 0);
//    cudaHostRegister(img_ir, signal_len * sizeof(uchar), 0);

//    // transfer from host to device
//    auto err = cudaMemcpy2DAsync(Image_V_Gr, step_G, img_v, IP_Img_1.width() * sizeof(uchar), IP_Img_1.width() * sizeof(uchar), IP_Img_1.height(), cudaMemcpyHostToDevice,stream1);
//    assert(err==0);
//    err = cudaMemcpy2DAsync(Image_IR_Gr, step_G, img_ir, IP_Img_1.width() * sizeof(uchar), IP_Img_1.width() * sizeof(uchar), IP_Img_1.height(), cudaMemcpyHostToDevice,stream2);
//    assert(err==0);

//    // call transform 8bit uint to 32bit float for both images
//    auto nperr = nppiConvert_8u32f_C1R(Image_V_Gr, step_G, Image_V_F, step_F, ROI);
//    assert(nperr==0);
//    nperr = nppiConvert_8u32f_C1R(Image_IR_Gr, step_G, Image_IR_F, step_F, ROI);
//    assert(nperr==0);
//    // copy from Npp object to cuda object
//    err = cudaMemcpy2DAsync(d_input_v, IP_Img_1.width() * sizeof(float), (float*)Image_V_F, step_F, IP_Img_1.width() * sizeof(float), IP_Img_1.height(), cudaMemcpyDeviceToDevice,stream1);
//    assert(err==0);
//    err = cudaMemcpy2DAsync(d_input_ir, IP_Img_1.width() * sizeof(float), (float*)Image_IR_F, step_F, IP_Img_1.width() * sizeof(float), IP_Img_1.height(), cudaMemcpyDeviceToDevice,stream2);
//    assert(err==0);

//    if (conv_methd==ePyramidConvolution::Gauss_separable)
//    {
//        Laplacian_const_Sep();
//        Image_reconst_Sep();
//    }
//    else
//    {
//        Laplacian_const_Rec();
//        Image_reconst_Rec();
//    }

//    return fused;
//}

//float* f_Lap_pyr::Fuse_Lap_Pyr_RGB(const QImage& IP_Img_1, const QImage& IP_Img_2) {

//    uchar4 *img_v, *img_ir;

//    img_v = (uchar4*)IP_Img_1.bits();
//    img_ir = (uchar4*)IP_Img_2.bits();

//    //Host Pageable memory
//    cudaHostRegister(img_v, signal_len * sizeof(uchar4), 0);
//    cudaHostRegister(img_ir, signal_len * sizeof(uchar4), 0);

//    cudaMemcpy2DAsync(Image_V_C, step_C, img_v, IP_Img_1.width() * sizeof(uchar4), IP_Img_1.width() * sizeof(uchar4), IP_Img_1.height(), cudaMemcpyHostToDevice, stream1);
//    cudaMemcpy2DAsync(Image_IR_C, step_C, img_ir, IP_Img_1.width() * sizeof(uchar4), IP_Img_1.width() * sizeof(uchar4), IP_Img_1.height(), cudaMemcpyHostToDevice, stream2);

//    nppiColorToGray_8u_C4C1R(Image_V_C, step_C, Image_V_Gr, step_G, ROI, coef);
//    nppiColorToGray_8u_C4C1R(Image_IR_C, step_C, Image_IR_Gr, step_G, ROI, coef);

//    nppiConvert_8u32f_C1R(Image_V_Gr, step_G, Image_V_F, step_F, ROI);
//    nppiConvert_8u32f_C1R(Image_IR_Gr, step_G, Image_IR_F, step_F, ROI);

//    cudaMemcpy2DAsync(d_input_v, IP_Img_1.width() * sizeof(float), (float*)Image_V_F, step_F, IP_Img_1.width() * sizeof(float), IP_Img_1.height(), cudaMemcpyDeviceToDevice, stream1);
//    cudaMemcpy2DAsync(d_input_ir, IP_Img_1.width() * sizeof(float), (float*)Image_IR_F, step_F, IP_Img_1.width() * sizeof(float), IP_Img_1.height(), cudaMemcpyDeviceToDevice, stream2);

//    if (conv_methd==ePyramidConvolution::Gauss_separable){
//        Laplacian_const_Sep();
//        Image_reconst_Sep();
//    }

//    else {
//        Laplacian_const_Rec();
//        Image_reconst_Rec();
//    }

//    return fused;
//}

float* f_Lap_pyr::Fuse_Lap_Pyr_Grayscale(const std::uint8_t* Img_visual, const std::uint8_t* Img_IR, const uint& width, const uint& height)
{
    std::uint8_t *img_v, *img_ir;
    img_v = (std::uint8_t*)Img_visual;
    img_ir = (std::uint8_t*)Img_IR;

    cudaHostRegister(img_v, signal_len * sizeof(float), 0);
    cudaHostRegister(img_ir, signal_len * sizeof(float), 0);

    auto src_pitch = width * sizeof(float);
    auto dst_pitch = width * sizeof(float);

    // transfer from host to device
    auto err = cudaMemcpy2DAsync(d_input_v, dst_pitch ,(void*)img_v,src_pitch , width * sizeof(float), height, cudaMemcpyHostToDevice,stream1);
    if(err!=cudaSuccess){
        std::cout<<"Some error: "<<err<<std::endl;
        std::exit(-1);
    }

    err = cudaMemcpy2DAsync(d_input_ir, dst_pitch, (void*)img_ir, src_pitch, width * sizeof(float), height, cudaMemcpyHostToDevice,stream2);
    if(err!=cudaSuccess){
        std::cout<<"Some error: "<<err<<std::endl;
        std::exit(-1);
    }

    if (conv_methd==ePyramidConvolution::Gauss_separable)
    {
        Laplacian_const_Sep();
        Image_reconst_Sep();
    }
    else
    {
        Laplacian_const_Rec();
        Image_reconst_Rec();
    }

    return fused;
}

float* f_Lap_pyr::Fuse_Lap_Pyr_RGBA(const std::uint8_t *img_visual, const std::uint8_t* img_IR,const uint& width, const uint& height) {

    uchar4 *img_v, *img_ir;

    img_v = (uchar4*)img_visual;
    img_ir = (uchar4*)img_IR;

    //Host Pageable memory
    cudaHostRegister(img_v, signal_len * sizeof(uchar4), 0);
    cudaHostRegister(img_ir, signal_len * sizeof(uchar4), 0);

    cudaMemcpy2DAsync(Image_V_C, step_C, img_v, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice, stream1);
    cudaMemcpy2DAsync(Image_IR_C, step_C, img_ir, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice, stream2);

    nppiColorToGray_8u_C4C1R(Image_V_C, step_C, Image_V_Gr, step_G, ROI, coef);
    nppiColorToGray_8u_C4C1R(Image_IR_C, step_C, Image_IR_Gr, step_G, ROI, coef);

    nppiConvert_8u32f_C1R(Image_V_Gr, step_G, Image_V_F, step_F, ROI);
    nppiConvert_8u32f_C1R(Image_IR_Gr, step_G, Image_IR_F, step_F, ROI);

    cudaMemcpy2DAsync(d_input_v, width * sizeof(float), (float*)Image_V_F, step_F, width * sizeof(float), height, cudaMemcpyDeviceToDevice, stream1);
    cudaMemcpy2DAsync(d_input_ir, width * sizeof(float), (float*)Image_IR_F, step_F, width * sizeof(float), height, cudaMemcpyDeviceToDevice, stream2);

    if (conv_methd==ePyramidConvolution::Gauss_separable){
        Laplacian_const_Sep();
        Image_reconst_Sep();
    }

    else {
        Laplacian_const_Rec();
        Image_reconst_Rec();
    }

    return fused;
}

void f_Lap_pyr::LP_free()
{

    dev_free();
    filt_free();

    cudaFree(d_input_ir); cudaFree(d_reduce_ir); cudaFree(d_expand_ir); cudaFree(d_buffer_v);
    cudaFree(d_input_v); cudaFree(d_reduce_v); cudaFree(d_expand_v); cudaFree(d_buffer_ir);

    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);

    cudaFreeHost(fused);

    free(length);
}

void f_Lap_pyr::dev_free() {

    cudaFree(d_lap_v);
    cudaFree(d_lap_ir);

}

void f_Lap_pyr ::reset_n_lev(int n_lev) {

    dev_free();

    set_n_lev(n_lev);
}
